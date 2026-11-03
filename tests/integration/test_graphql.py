"""GraphQL API tests (mock mode, no network): parity with REST, declines, errors."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from arag.serve.api import create_app

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "config.yaml"
OUT_OF_SCOPE = "How do I enable HTTP/2 support in Breeze?"
ASK = """query($q: String!) {
  ask(question: $q) { answer declined citations { source snippet } trace { stage latencyMs } }
}"""


def _client(config=CONFIG):
    # raise_server_exceptions=False so a 500 shows up as a status code, not an exception.
    return TestClient(create_app(str(config)), raise_server_exceptions=False)


def _ask(c, question):
    r = c.post("/graphql", json={"query": ASK, "variables": {"q": question}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "errors" not in body, body
    return body["data"]["ask"]


def test_rest_and_graphql_return_same_answer():
    q = "How do I define a path parameter?"
    with _client() as c:
        rest = c.post("/query", json={"query": q}).json()
        gql = _ask(c, q)
    assert rest["citations"], "need at least one citation for a meaningful comparison"
    assert gql["answer"] == rest["answer"]
    assert gql["citations"] == [
        {"source": x["doc_id"], "snippet": x["quote"]} for x in rest["citations"]
    ]
    assert gql["declined"] is rest["abstained"] is False
    assert [t["stage"] for t in gql["trace"]] == [t["stage"] for t in rest["trace"]]


def test_declined_question_sets_declined(tmp_path):
    # ask() has no flags argument, so turn the agent + CRAG on in the server's config.
    raw = yaml.safe_load(CONFIG.read_text())
    raw["agent"]["enabled"] = True
    raw["agent"]["crag"]["enabled"] = True
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    with _client(cfg) as c:
        gql = _ask(c, OUT_OF_SCOPE)
        rest = c.post("/query", json={"query": OUT_OF_SCOPE}).json()
    assert gql["declined"] is True
    assert rest["abstained"] is True
    assert gql["answer"] == rest["answer"]


@pytest.mark.parametrize(
    "body",
    [
        {"query": "{ nope }"},                          # unknown field
        {"query": "{ ask { answer } }"},                # missing required argument
        {"query": "{ ask(question: "},                  # syntax error
        {"query": "{ ask(question: 1) { answer } }"},   # wrong argument type
        {"query": "{ documents(limit: -1) { id } }"},   # rejected by the resolver
    ],
)
def test_invalid_query_returns_graphql_error_not_500(body):
    with _client() as c:
        r = c.post("/graphql", json=body)
    assert r.status_code != 500
    assert r.json()["errors"][0]["message"]


def test_health_and_documents():
    with _client() as c:
        data = c.post("/graphql", json={"query": "{ health documents(limit: 2) { id title } }"}).json()["data"]
        corpus = c.get("/corpus").json()["docs"]
    assert data["health"] == "ok"
    assert data["documents"] == [{"id": d["doc_id"], "title": d["title"]} for d in corpus[:2]]
