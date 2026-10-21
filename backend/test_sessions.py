import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from llama_index.core import Document
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateSchema, DropSchema

import auth
import ingest
import main
import vector_db


@contextmanager
def isolated_database():
    engine = create_engine(URL.create(
        "postgresql+psycopg",
        username=os.getenv("POSTGRES_USER", "docusense"),
        password=os.getenv("POSTGRES_PASSWORD", "local-dev-only"),
        host=os.getenv("POSTGRES_TEST_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_TEST_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "docusense"),
    ))
    schema = "test_" + uuid4().hex
    try:
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        test_engine = engine.execution_options(schema_translate_map={None: schema})
        with patch.object(auth, "database_engine", return_value=test_engine):
            yield
    finally:
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        engine.dispose()


class SessionIsolationTest(unittest.TestCase):
    def test_auth_sessions_require_existing_user(self):
        with isolated_database():
            auth.initialize_database()
            with self.assertRaises(IntegrityError):
                auth.create_auth_session("missing-user")

    def test_claim_failure_can_be_retried_at_login(self):
        guest_id = "a" * 32
        document_id = "b" * 32
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, "UPLOAD_DIR", Path(directory) / "uploads"), \
            isolated_database(), \
                patch.object(main, "claim_document_vectors", side_effect=[RuntimeError("Qdrant unavailable"), None]):
            auth.initialize_database()
            document_dir = main.UPLOAD_DIR / guest_id / document_id
            document_dir.mkdir(parents=True)
            (document_dir / "paper.pdf").write_bytes(b"%PDF-1.4")
            guest = TestClient(main.app, cookies={main.SESSION_COOKIE: guest_id}, raise_server_exceptions=False)
            credentials = {"email": "retry@example.com", "password": "strong-password"}
            self.assertEqual(guest.post("/api/v1/auth/register", json=credentials).status_code, 500)
            self.assertIsNone(guest.get("/api/v1/auth/me").json()["user"])
            self.assertIsNotNone(auth.document_for_owner(document_id, guest_id, None))
            self.assertEqual(guest.post("/api/v1/auth/login", json=credentials).status_code, 200)
            self.assertEqual(guest.get("/api/v1/documents").json()[0]["id"], document_id)

    def test_existing_guest_pdf_is_claimed_without_database_record(self):
        guest_id = "a" * 32
        document_id = "b" * 32
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, "UPLOAD_DIR", Path(directory) / "uploads"), \
            isolated_database(), \
                patch.object(main, "claim_document_vectors") as claim_vectors:
            auth.initialize_database()
            document_dir = main.UPLOAD_DIR / guest_id / document_id
            document_dir.mkdir(parents=True)
            (document_dir / "old.PDF").write_bytes(b"%PDF-1.4")
            guest = TestClient(main.app, cookies={main.SESSION_COOKIE: guest_id})
            registered = guest.post(
                "/api/v1/auth/register",
                json={"email": "legacy@example.com", "password": "strong-password"},
            )
            self.assertEqual(registered.status_code, 201, registered.text)
            user_id = registered.json()["user"]["id"]
            claim_vectors.assert_called_once_with(document_id, guest_id, user_id)
            self.assertEqual(guest.get("/api/v1/documents").json(), [{"id": document_id, "filename": "old.PDF"}])

    def test_guest_upload_is_claimed_on_register_and_available_after_login(self):
        class FakeIndex:
            def __init__(self):
                self.filters = []

            def as_query_engine(self, **kwargs):
                metadata_filter = kwargs["filters"].filters[0]
                self.filters.append((metadata_filter.key, metadata_filter.value))
                return type("Engine", (), {"query": lambda self, question: type(
                    "Result", (), {"response": "ok", "source_nodes": []}
                )()})()

        index = FakeIndex()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, "UPLOAD_DIR", Path(directory) / "uploads"), \
            isolated_database(), \
                patch.dict(main.app_state, {"index": index}), \
                patch.object(main, "add_document_to_vector_database", return_value=1), \
                patch.object(main, "claim_document_vectors") as claim_vectors, \
                patch.object(main, "delete_document_from_vector_database") as delete_vectors:
            auth.initialize_database()
            guest = TestClient(main.app)
            upload = guest.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF-1.4\ntest", "application/pdf")},
            )
            self.assertEqual(upload.status_code, 201)
            document_id = upload.json()["document_id"]
            old_guest_id = guest.cookies.get(main.SESSION_COOKIE)
            old_guest = TestClient(main.app, cookies={main.SESSION_COOKIE: old_guest_id})
            credentials = {"email": "Alice@Example.com", "password": "a-strong-password"}
            registered = guest.post("/api/v1/auth/register", json=credentials)
            self.assertEqual(registered.status_code, 201, registered.text)
            user_id = registered.json()["user"]["id"]
            claim_vectors.assert_called_once_with(document_id, old_guest_id, user_id)
            self.assertNotEqual(guest.cookies.get(main.SESSION_COOKIE), old_guest_id)
            self.assertEqual(guest.get("/api/v1/documents").json(), [{"id": document_id, "filename": "paper.pdf"}])
            self.assertEqual(old_guest.get("/api/v1/documents").json(), [])
            self.assertEqual(old_guest.delete(f"/api/v1/documents/{document_id}").status_code, 404)

            self.assertEqual(guest.post("/api/v1/query", json={"question": "hi"}).status_code, 200)
            self.assertEqual(index.filters[-1], ("user_id", user_id))
            old_token = guest.cookies.get(main.AUTH_COOKIE)
            self.assertEqual(guest.post("/api/v1/auth/logout").status_code, 200)
            self.assertIsNone(guest.get("/api/v1/auth/me").json()["user"])
            self.assertEqual(guest.get("/api/v1/documents").json(), [])
            revoked = TestClient(main.app, cookies={main.AUTH_COOKIE: old_token})
            self.assertIsNone(revoked.get("/api/v1/auth/me").json()["user"])

            other_browser = TestClient(main.app)
            self.assertEqual(other_browser.get("/api/v1/documents").json(), [])
            stranger = TestClient(main.app)
            self.assertEqual(stranger.post(
                "/api/v1/auth/register",
                json={"email": "stranger@example.com", "password": "another-password"},
            ).status_code, 201)
            self.assertEqual(stranger.get("/api/v1/documents").json(), [])
            self.assertEqual(stranger.delete(f"/api/v1/documents/{document_id}").status_code, 404)
            login = other_browser.post("/api/v1/auth/login", json=credentials)
            self.assertEqual(login.status_code, 200, login.text)
            self.assertEqual(other_browser.get("/api/v1/documents").json()[0]["id"], document_id)
            self.assertEqual(other_browser.delete(f"/api/v1/documents/{document_id}").status_code, 200)
            delete_vectors.assert_called_once_with(document_id, user_id, "user_id")

    def test_claim_updates_only_matching_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            database = QdrantClient(path=directory)
            database.create_collection("documents", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
            database.upsert("documents", points=[
                PointStruct(id=1, vector=[1.0, 0.0], payload={"doc_id": "first", "session_id": "guest"}),
                PointStruct(id=2, vector=[0.0, 1.0], payload={"doc_id": "second", "session_id": "guest"}),
            ])
            database.close()

            with patch.object(vector_db, "QdrantClient", side_effect=lambda **kwargs: QdrantClient(path=directory)):
                vector_db.claim_document_vectors("first", "guest", "account")
                database = QdrantClient(path=directory)
                first, second = database.retrieve("documents", ids=[1, 2], with_payload=True)
                self.assertEqual(first.payload["user_id"], "account")
                self.assertNotIn("session_id", first.payload)
                self.assertEqual(second.payload["session_id"], "guest")
                database.close()
                vector_db.delete_document_from_vector_database("first", "account", "user_id")

            database = QdrantClient(path=directory)
            self.assertEqual([point.id for point in database.scroll("documents")[0]], [2])
            database.close()

    def test_upload_query_and_delete_are_scoped_to_session(self):
        class FakeQueryEngine:
            def query(self, question):
                return type("Response", (), {
                    "response": "ok",
                    "response_gen": iter(["ok"]),
                    "source_nodes": [],
                })()

        class FakeIndex:
            def __init__(self):
                self.filters = []

            def as_query_engine(self, **kwargs):
                self.filters.append(kwargs["filters"].filters[0].value)
                return FakeQueryEngine()

        index = FakeIndex()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, "UPLOAD_DIR", Path(directory)), \
            isolated_database(), \
                patch.dict(main.app_state, {"index": index}), \
                patch.object(main, "add_document_to_vector_database", return_value=1) as add_document, \
                patch.object(main, "delete_document_from_vector_database") as delete_document:
            auth.initialize_database()
            first = TestClient(main.app)
            second = TestClient(main.app)
            upload = first.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF-1.4\ntest", "application/pdf")},
                headers={"Origin": "http://localhost:5173"},
            )
            self.assertEqual(upload.status_code, 201)
            self.assertEqual(upload.headers["access-control-allow-credentials"], "true")
            self.assertEqual(upload.headers["access-control-allow-origin"], "http://localhost:5173")
            rejected = first.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
                headers={"Origin": "http://untrusted.example"},
            )
            self.assertEqual(rejected.status_code, 403)
            document_id = upload.json()["document_id"]
            first_session = first.cookies.get(main.SESSION_COOKIE)
            self.assertEqual(add_document.call_args.args[2], first_session)
            self.assertEqual(add_document.call_args.args[3], "session_id")

            self.assertEqual(first.post("/api/v1/query", json={"question": "hello"}).status_code, 200)
            self.assertEqual(second.post("/api/v1/query", json={"question": "hello"}).status_code, 200)
            stream = first.post("/api/v1/query/stream", json={"question": "hello"})
            self.assertEqual(stream.status_code, 200)
            self.assertIn("event: delta", stream.text)
            second_session = second.cookies.get(main.SESSION_COOKIE)
            self.assertNotEqual(first_session, second_session)
            self.assertEqual(index.filters, [first_session, second_session, first_session])

            self.assertEqual(second.delete(f"/api/v1/documents/{document_id}").status_code, 404)
            delete_document.assert_not_called()
            self.assertEqual(first.delete(f"/api/v1/documents/{document_id}").status_code, 200)
            delete_document.assert_called_once_with(document_id, first_session, "session_id")
            self.assertFalse((Path(directory) / first_session / document_id).exists())

    def test_ingested_chunks_include_session_and_document_ids(self):
        class FakeReader:
            def __init__(self, input_files):
                pass

            def load_data(self):
                return [Document(text="A sample PDF paragraph.")]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4")
            with patch.object(ingest, "SimpleDirectoryReader", FakeReader):
                nodes = ingest.load_and_chunk_documents(str(path), "document-1", "session-1")

        payload = node_to_metadata_dict(nodes[0])
        self.assertEqual(payload["doc_id"], "document-1")
        self.assertEqual(payload["session_id"], "session-1")


if __name__ == "__main__":
    unittest.main()
