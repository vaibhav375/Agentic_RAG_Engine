"""GraphQL API (Strawberry), mounted at /graphql next to the REST API.

Resolvers only reshape data. Every answer comes from the same `run_query` the
REST /query route calls, so the two surfaces cannot drift apart.

Nullability follows the requested contract: the top-level fields (`ask`,
`health`, `documents`) are nullable, so one failing field doesn't null the
whole response.
"""

import asyncio

import strawberry
from strawberry.fastapi import GraphQLRouter


@strawberry.type
class Citation:
    source: str  # doc_id: the same id `documents` returns
    snippet: str | None


@strawberry.type
class TraceStage:
    stage: str
    latency_ms: float


@strawberry.type
class Answer:
    answer: str
    citations: list[Citation]
    declined: bool
    trace: list[TraceStage]


@strawberry.type
class Document:
    id: str
    title: str


@strawberry.type
class Query:
    @strawberry.field
    async def ask(self, info: strawberry.Info, question: str) -> Answer | None:
        # REST /query is a sync route, so FastAPI runs it in a thread. Do the
        # same here so a slow pipeline doesn't block the event loop.
        r = await asyncio.to_thread(info.context["run_query"], question)
        return Answer(
            answer=r.answer,
            citations=[Citation(source=c["doc_id"], snippet=c.get("quote")) for c in r.citations],
            declined=r.abstained,
            trace=[TraceStage(stage=t["stage"], latency_ms=t["ms"]) for t in r.trace],
        )

    @strawberry.field
    def health(self, info: strawberry.Info) -> str | None:
        return info.context["health"]()

    @strawberry.field
    def documents(self, info: strawberry.Info, limit: int = 20) -> list[Document | None] | None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        docs = info.context["list_documents"]()
        return [Document(id=k, title=docs[k]["title"]) for k in sorted(docs)[:limit]]


schema = strawberry.Schema(query=Query)


def graphql_router(services: dict) -> GraphQLRouter:
    """`services` holds the callables the resolvers use: run_query, health, list_documents."""

    async def context() -> dict:
        return services

    return GraphQLRouter(schema, context_getter=context)
