import asyncio
import importlib
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


class RootRuns(BaseCallbackHandler):
    def __init__(self):
        self.runs = []

    def on_chain_start(self, serialized, inputs, *, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            self.runs.append(kwargs)


class TracingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dotenv = patch("dotenv.load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        with patch("langsmith.Client.pull_agent", side_effect=RuntimeError("offline")):
            self.module = importlib.import_module("concierge.graph")
        self.model = patch.object(
            self.module,
            "_make_model",
            return_value=Mock(invoke=Mock(return_value=AIMessage(content="Hello"))),
        )
        self.model.start()
        self.addCleanup(self.model.stop)

    def graph_for_environment(self, environment):
        if environment is None:
            os.environ.pop("APP_ENV", None)
        else:
            os.environ["APP_ENV"] = environment
        with patch.object(self.module, "get_prompt", return_value="Test prompt"):
            self.module = importlib.reload(self.module)
        self.model.stop()
        self.model = patch.object(
            self.module,
            "_make_model",
            return_value=Mock(invoke=Mock(return_value=AIMessage(content="Hello"))),
        )
        self.model.start()
        self.addCleanup(self.model.stop)
        return self.module.graph

    def test_root_name_and_environment_for_all_invocation_paths(self):
        for environment in ("staging", None):
            graph = self.graph_for_environment(environment)
            self.assertEqual(self.module._build_graph().name, "banking_concierge")
            for mode in ("invoke", "ainvoke", "stream", "astream"):
                with self.subTest(environment=environment, mode=mode):
                    roots = RootRuns()
                    useful = {"revision_id": "revision", "ls_integration": "test"}
                    config = {
                        "callbacks": [roots],
                        "configurable": {"thread_id": "thread"},
                        "metadata": useful,
                    }
                    inputs = {"messages": [("user", "Hello")]}
                    if mode == "invoke":
                        graph.invoke(inputs, config)
                    elif mode == "stream":
                        list(graph.stream(inputs, config))
                    elif mode == "ainvoke":
                        asyncio.run(graph.ainvoke(inputs, config))
                    else:

                        async def consume(graph=graph, inputs=inputs, config=config):
                            return [
                                chunk async for chunk in graph.astream(inputs, config)
                            ]

                        asyncio.run(consume())
                    self.assertEqual(len(roots.runs), 1)
                    root = roots.runs[0]
                    self.assertEqual(root["name"], "banking_concierge")
                    self.assertEqual(
                        root["metadata"]["environment"], environment or "development"
                    )
                    self.assertEqual(root["metadata"]["thread_id"], "thread")
                    for key, value in useful.items():
                        self.assertEqual(root["metadata"][key], value)

    def test_deployment_root_preserves_metadata(self):
        from langgraph.graph.state import CompiledStateGraph

        self.graph_for_environment("production")
        for metadata in (
            {"revision_id": "deployed"},
            {"environment": "loadgen", "scenario": "healthy_faq"},
        ):
            roots = RootRuns()
            config = {
                "callbacks": [roots],
                "metadata": metadata,
                "configurable": {"thread_id": "deployed-thread"},
            }
            graph = self.module.deployment_graph(config)
            self.assertIsInstance(graph, CompiledStateGraph)
            asyncio.run(graph.ainvoke({"messages": [("user", "Hello")]}, config))
            self.assertEqual(len(roots.runs), 1)
            root = roots.runs[0]
            self.assertEqual(root["name"], "banking_concierge")
            self.assertEqual(
                root["metadata"]["environment"],
                metadata.get("environment", "production"),
            )
            for key, value in metadata.items():
                self.assertEqual(root["metadata"][key], value)

    def test_all_loadgen_scenarios_local_and_remote(self):
        from scripts import load_generation

        graph = self.graph_for_environment("production")
        categories = sorted({p.category for p in load_generation.ALL_PROMPTS})
        prompts = [
            load_generation.Prompt(category, ("Hello", "Again"))
            for category in categories
        ]
        for mode in ("local", "remote"):
            with self.subTest(mode=mode):
                roots = RootRuns()
                traced_graph = graph.with_config(callbacks=[roots])
                if mode == "local":
                    with (
                        patch.object(self.module, "graph", traced_graph),
                        redirect_stdout(io.StringIO()),
                    ):
                        asyncio.run(load_generation._run_local(prompts, "", 1, 1, 1))
                else:

                    async def wait(
                        thread_id,
                        assistant,
                        *,
                        input,
                        config,
                        metadata,
                        traced_graph=traced_graph,
                    ):
                        return await traced_graph.ainvoke(
                            input,
                            config={
                                **config,
                                "metadata": metadata,
                                "configurable": {"thread_id": thread_id},
                            },
                        )

                    client = Mock()
                    client.threads.create = AsyncMock(
                        return_value={"thread_id": "remote-thread"}
                    )
                    client.runs.wait = AsyncMock(side_effect=wait)
                    with (
                        patch("langgraph_sdk.get_client", return_value=client),
                        redirect_stdout(io.StringIO()),
                    ):
                        asyncio.run(
                            load_generation._run_remote(
                                prompts, "https://example.test", "", "agent", 1, 1, 1
                            )
                        )
                self.assertEqual(len(roots.runs), len(prompts) * 2)
                for prompt, pair in zip(
                    prompts, zip(roots.runs[::2], roots.runs[1::2])
                ):
                    for root in pair:
                        self.assertEqual(root["name"], "banking_concierge")
                        self.assertIn("loadgen", root["tags"])
                        self.assertIn(f"category:{prompt.category}", root["tags"])
                        self.assertEqual(root["metadata"]["environment"], "loadgen")
                        self.assertEqual(root["metadata"]["scenario"], prompt.category)
                        self.assertTrue(root["metadata"]["thread_id"])


if __name__ == "__main__":
    unittest.main()
