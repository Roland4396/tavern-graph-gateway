from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from gateway import DiskCache, GraphGateway, PromptJournal


class DiskCachePolicyTest(unittest.TestCase):
    def test_extension_assets_are_cached_with_a_bounded_lifetime(self) -> None:
        for path in [
            "/extensions/was-node-suite-comfyui/vendor/three/three.core.js",
            "/extensions/was-node-suite-comfyui/viewer/views/code_scripts/prism.min.txt",
            "/rgthree/common/utils.js", "/scripts/app.js", "/kjweb_async/widget.js",
        ]:
            with self.subTest(path=path):
                self.assertEqual(DiskCache.policy(path), "private, max-age=300")
                self.assertEqual(DiskCache.ttl(path + "?v=1"), 86400)

    def test_dynamic_data_and_extension_catalog_are_not_cached(self) -> None:
        for path in [
            "/extensions", "/api/extensions", "/rgthree/config.js",
            "/api/object_info", "/queue", "/view?filename=output.png",
            "/api/userdata/workflows/test.json", "/extensions/foo/settings.json",
        ]:
            with self.subTest(path=path):
                self.assertIsNone(DiskCache.policy(path))

    def test_content_addressed_global_subgraph_is_immutable(self) -> None:
        path = "/api/global_subgraphs/" + "a" * 64
        self.assertEqual(
            DiskCache.policy(path),
            "private, max-age=31536000, immutable",
        )

    def test_mutable_subgraph_catalog_is_not_cached(self) -> None:
        self.assertIsNone(DiskCache.policy("/api/global_subgraphs"))
        self.assertIsNone(DiskCache.policy("/api/userdata?dir=subgraphs"))


class GenerationRequestTest(unittest.TestCase):
    def test_both_free_routes_trigger_cleanup_without_waking_graph(self) -> None:
        for path in ("/free", "/api/free", "/api/free/"):
            self.assertTrue(GraphGateway.is_free_request("POST", path))
            self.assertFalse(GraphGateway.is_free_request("GET", path))
            self.assertFalse(GraphGateway.is_generation_request("POST", path))

    def test_legacy_prompt_route_wakes_graph(self) -> None:
        self.assertTrue(GraphGateway.is_generation_request("POST", "/prompt"))

    def test_current_api_prompt_route_wakes_graph(self) -> None:
        self.assertTrue(GraphGateway.is_generation_request("POST", "/api/prompt"))

    def test_trailing_slash_is_accepted(self) -> None:
        self.assertTrue(GraphGateway.is_generation_request("POST", "/api/prompt/"))

    def test_non_generation_routes_do_not_wake_graph(self) -> None:
        self.assertFalse(GraphGateway.is_generation_request("GET", "/api/prompt"))
        self.assertFalse(
            GraphGateway.is_generation_request("POST", "/api/userdata/workflows/test.json")
        )


class PromptJournalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "prompts.sqlite3"
        self.journal = PromptJournal(self.path)

    def tearDown(self) -> None:
        self.journal.close()
        self.temporary.cleanup()

    @staticmethod
    def request() -> bytes:
        return json.dumps(
            {
                "client_id": "browser-1",
                "prompt": {
                    "4": {
                        "class_type": "MiniMaxH3FL2VASampler",
                        "inputs": {"steps": 50},
                    }
                },
            }
        ).encode()

    def test_injects_stable_operation_metadata_and_deduplicates(self) -> None:
        first, created = self.journal.prepare(self.request())
        duplicate, duplicate_created = self.journal.prepare(self.request())

        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(first["prompt_id"], duplicate["prompt_id"])
        self.assertEqual(len(duplicate["operations"]), 1)
        operation = first["operations"][0]
        inputs = first["request"]["prompt"]["4"]["inputs"]
        self.assertEqual(inputs["_h3_operation_id"], operation["operation_id"])
        self.assertEqual(inputs["_graph_prompt_id"], first["prompt_id"])
        self.assertEqual(inputs["_graph_node_id"], "4")

        self.journal.close()
        self.journal = PromptJournal(self.path)
        recovered = self.journal.get(first["prompt_id"])
        assert recovered is not None
        self.assertEqual(
            recovered["request"]["prompt"]["4"]["inputs"]["_h3_operation_id"],
            operation["operation_id"],
        )

    def test_forward_claim_prevents_concurrent_replay_but_allows_retry(self) -> None:
        prompt, _created = self.journal.prepare(self.request())
        prompt_id = prompt["prompt_id"]
        self.assertTrue(self.journal.claim_forward(prompt_id))
        self.assertFalse(self.journal.claim_forward(prompt_id))
        self.assertTrue(self.journal.claim_forward(prompt_id, retry_after=0))

    def test_cancel_must_be_confirmed_before_operation_ack(self) -> None:
        prompt, _created = self.journal.prepare(self.request())
        prompt_id = prompt["prompt_id"]
        operation_id = prompt["operations"][0]["operation_id"]
        pending = self.journal.request_cancel([prompt_id], reason={"test": True})

        self.assertEqual([item["operation_id"] for item in pending], [operation_id])
        self.assertEqual(self.journal.pending_ack_prompts(), [])
        self.journal.mark_cancel_confirmed(operation_id)
        self.assertEqual(self.journal.pending_ack_prompts(), [prompt_id])


class PromptRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_replay_and_ack_use_durable_operation_id(self) -> None:
        received: dict[str, object] = {}

        async def prompt(request: web.Request) -> web.Response:
            received["prompt"] = await request.json()
            return web.json_response({"prompt_id": received["prompt"]["prompt_id"]})

        async def ack(request: web.Request) -> web.Response:
            received["ack"] = await request.json()
            return web.json_response({"status": "ok", "acknowledged": 1})

        app = web.Application()
        app.router.add_post("/prompt", prompt)
        app.router.add_post("/h3-control/v1/operations/ack", ack)
        with tempfile.TemporaryDirectory() as temporary:
            async with TestServer(app) as server:
                journal = PromptJournal(Path(temporary) / "prompts.sqlite3")
                record, _created = journal.prepare(PromptJournalTest.request())
                gateway = GraphGateway.__new__(GraphGateway)
                gateway.journal = journal
                gateway.service_reachable = True
                gateway.session = ClientSession()
                try:
                    with patch(
                        "gateway.UPSTREAM_HTTP", str(server.make_url("")).rstrip("/")
                    ), patch(
                        "gateway.H3_CONTROL_URL",
                        str(server.make_url("/h3-control")).rstrip("/"),
                    ):
                        self.assertTrue(await gateway.forward_journal_prompt(record))
                        journal.mark_terminal(record["prompt_id"], "completed")
                        await gateway.acknowledge_prompt_operations(record["prompt_id"])
                finally:
                    await gateway.session.close()
                    journal.close()

        operation_id = record["operations"][0]["operation_id"]
        self.assertEqual(
            received["prompt"]["prompt"]["4"]["inputs"]["_h3_operation_id"],
            operation_id,
        )
        self.assertEqual(received["ack"], {"operation_ids": [operation_id]})

    async def test_cancellation_retries_until_h3_is_terminal(self) -> None:
        calls = 0

        async def cancel(request: web.Request) -> web.Response:
            nonlocal calls
            calls += 1
            operation_id = request.match_info["operation_id"]
            status = "cancel_requested" if calls == 1 else "cancelled"
            return web.json_response({"id": operation_id, "status": status}, status=202 if calls == 1 else 200)

        app = web.Application()
        app.router.add_delete("/h3/v1/videos/{operation_id}", cancel)
        with tempfile.TemporaryDirectory() as temporary:
            async with TestServer(app) as server:
                journal = PromptJournal(Path(temporary) / "prompts.sqlite3")
                record, _created = journal.prepare(PromptJournalTest.request())
                operation = journal.request_cancel(
                    [record["prompt_id"]], reason={"test": True}
                )[0]
                gateway = GraphGateway.__new__(GraphGateway)
                gateway.journal = journal
                gateway.session = ClientSession()
                try:
                    with patch(
                        "gateway.H3_CONTROL_URL",
                        str(server.make_url("/h3-control")).rstrip("/"),
                    ):
                        self.assertFalse(await gateway.cancel_h3_operation(operation))
                        self.assertEqual(journal.pending_ack_prompts(), [])
                        self.assertTrue(await gateway.cancel_h3_operation(operation))
                        self.assertEqual(
                            journal.pending_ack_prompts(), [record["prompt_id"]]
                        )
                finally:
                    await gateway.session.close()
                    journal.close()

        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
