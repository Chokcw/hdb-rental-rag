"""Weaviate storage. Everything Weaviate-specific lives in this file.

The rest of the project only uses: ensure_collection / existing / upsert /
update_properties / delete / hybrid_search. To compare against another store
later (SQLite, pgvector...), write a class with the same methods.

Connection modes (WEAVIATE_MODE):
  embedded (default)  Weaviate runs inside this Python process. Linux/macOS only.
                      Data is kept in data/weaviate and survives restarts.
                      The server stops when the script exits.
  docker              connect to a Weaviate already running on localhost:8080
                      (docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate)
"""
from __future__ import annotations

import atexit
import os
from dataclasses import dataclass

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from embed import embedding_key
from tracing import trace

WEAVIATE_VERSION = os.getenv("WEAVIATE_VERSION", "1.40.2")  # pinned: the client's default may lag
DATA_DIR = os.getenv("WEAVIATE_DATA_DIR", "data/weaviate")
BINARY_DIR = os.getenv("WEAVIATE_BINARY_DIR", "data/weaviate-bin")


@dataclass
class Hit:
    chunk_id: str
    page_id: str
    title: str
    heading_path: str
    url: str
    text: str
    tier: int
    audience: str
    fetched_at: str
    score: float


def collection_name(mode: str) -> str:
    return f"HdbChunk_{embedding_key(mode)}"


def connect() -> weaviate.WeaviateClient:
    if os.getenv("WEAVIATE_MODE", "embedded").lower() == "docker":
        return weaviate.connect_to_local()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BINARY_DIR, exist_ok=True)
    return weaviate.connect_to_embedded(
        version=WEAVIATE_VERSION,
        persistence_data_path=DATA_DIR,
        binary_path=BINARY_DIR,
        # CLUSTER_ADVERTISE_ADDR: embedded Weaviate fails to start on machines with no private
        # network address (offline, some VPNs/containers); pinning it to localhost avoids that.
        environment_variables={"LOG_LEVEL": "error", "CLUSTER_ADVERTISE_ADDR": "127.0.0.1",
                               "DISABLE_TELEMETRY": "true"},
    )


class WeaviateStore:
    def __init__(self, mode: str):
        self.client = connect()
        self.name = collection_name(mode)
        atexit.register(self.close)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    # ------------------------------------------------------------ setup

    def ensure_collection(self, rebuild: bool = False):
        if rebuild and self.client.collections.exists(self.name):
            self.client.collections.delete(self.name)
        if self.client.collections.exists(self.name):
            return
        field = Tokenization.FIELD  # exact-match values, not split into words
        self.client.collections.create(
            self.name,
            vector_config=Configure.Vectors.self_provided(),  # we supply the vectors
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT, tokenization=field),
                Property(name="chunk_hash", data_type=DataType.TEXT, tokenization=field),
                Property(name="page_id", data_type=DataType.TEXT, tokenization=field),
                Property(name="audience", data_type=DataType.TEXT, tokenization=field),
                Property(name="source", data_type=DataType.TEXT, tokenization=field),
                Property(name="url", data_type=DataType.TEXT, tokenization=field, index_searchable=False),
                Property(name="fetched_at", data_type=DataType.TEXT, tokenization=field, index_searchable=False),
                Property(name="tier", data_type=DataType.INT),
                Property(name="position", data_type=DataType.INT),
                # the three fields keyword (BM25) search looks at:
                Property(name="title", data_type=DataType.TEXT),
                Property(name="heading_path", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
        )

    @property
    def col(self):
        return self.client.collections.get(self.name)

    # ------------------------------------------------------------ writes

    def existing(self) -> dict[str, str]:
        """uuid -> chunk_hash for everything already stored."""
        return {str(o.uuid): o.properties.get("chunk_hash", "")
                for o in self.col.iterator(return_properties=["chunk_hash"])}

    @staticmethod
    def _props(c) -> dict:
        return {
            "chunk_id": c.chunk_id, "chunk_hash": c.chunk_hash, "page_id": c.page_id,
            "audience": c.audience, "source": c.source, "url": c.url, "fetched_at": c.fetched_at,
            "tier": c.tier, "position": c.position, "title": c.title,
            "heading_path": c.heading_path, "text": c.text,
        }

    def upsert(self, chunks, vectors):
        with self.col.batch.dynamic() as batch:
            for c, v in zip(chunks, vectors):
                batch.add_object(properties=self._props(c), uuid=c.uuid, vector=v)
        failed = self.col.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} objects failed to import, first: {failed[0].message}")

    def update_properties(self, chunks):
        """Refresh metadata (e.g. fetched_at) without touching the stored vector."""
        for c in chunks:
            self.col.data.update(uuid=c.uuid, properties=self._props(c))

    def delete(self, uuids: list[str]):
        if uuids:
            self.col.data.delete_many(where=Filter.by_id().contains_any(uuids))

    def count(self) -> int:
        return self.col.aggregate.over_all(total_count=True).total_count

    # ------------------------------------------------------------ reads

    @trace("weaviate_hybrid_search", run_type="retriever")
    def hybrid_search(self, query: str, vector: list[float], k: int = 5,
                      alpha: float = 0.5, audience: str | None = None) -> list[dict]:
        """alpha: 0 = keyword only, 1 = vector only. Returns plain dicts (easy to trace)."""
        flt = None
        if audience:  # an owner question should see owner pages and pages for everyone
            flt = Filter.by_property("audience").contains_any([audience, "both"])
        res = self.col.query.hybrid(
            query=query, vector=vector, alpha=alpha, limit=k, filters=flt,
            query_properties=["heading_path^2", "title", "text"],
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=MetadataQuery(score=True),
        )
        hits = []
        for o in res.objects:
            p = o.properties
            hits.append(Hit(
                chunk_id=p["chunk_id"], page_id=p["page_id"], title=p["title"],
                heading_path=p["heading_path"], url=p["url"], text=p["text"],
                tier=p["tier"], audience=p["audience"], fetched_at=p["fetched_at"],
                score=round(o.metadata.score or 0.0, 4),
            ).__dict__)
        return hits
