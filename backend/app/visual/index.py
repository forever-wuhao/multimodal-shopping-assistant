"""Immutable FAISS + SQLite snapshots. Exact filtering precedes product ranking."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from contextlib import closing

import numpy as np

from .data import image_path, manifest_digest, read_image
from .encoder import normalize


def build_snapshot(rows, root, output, encoder, adapter=None, batch_size=16):
    import faiss
    rows = [row for row in rows if row.role == "gallery"]
    if not rows:
        raise ValueError("No gallery images")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new versioned snapshot directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".index-", dir=output.parent))
    try:
        embeddings = []
        (staging / "images").mkdir()
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = [read_image(image_path(root, row.image)) for row in batch]
            embeddings.append(encoder.images(images))
            for row, image in zip(batch, images):
                image.thumbnail((640, 640))
                image.save(staging / "images" / f"{row.image_id}.jpg", quality=85)
        vectors = normalize(np.concatenate(embeddings))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(staging / "vectors.faiss"))
        with closing(sqlite3.connect(staging / "catalog.sqlite3")) as connection:
            with connection:
                connection.execute("CREATE TABLE images (vector_id INTEGER PRIMARY KEY, image_id TEXT UNIQUE, product_id TEXT, document TEXT)")
                connection.executemany("INSERT INTO images VALUES (?, ?, ?, ?)", [(i, row.image_id, row.product_id, row.model_dump_json()) for i, row in enumerate(rows)])
        if adapter:
            shutil.copy2(adapter, staging / "adapter.npz")
        metadata = {"schema": 1, "encoder": encoder.identity(), "images": len(rows),
                    "products": len({r.product_id for r in rows}), "manifest_sha256": manifest_digest(rows),
                    "files": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest() for name in ("vectors.faiss", "catalog.sqlite3")}}
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.rename(staging, output)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class ProductIndex:
    def __init__(self, directory):
        import faiss
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema") != 1:
            raise ValueError("Unsupported index schema")
        for name in ("vectors.faiss", "catalog.sqlite3"):
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != self.metadata["files"][name]:
                raise ValueError("Index/catalog integrity mismatch; rebuild snapshot")
        self.index = faiss.read_index(str(self.directory / "vectors.faiss"))
        with closing(sqlite3.connect(self.directory / "catalog.sqlite3")) as connection:
            self.rows = [json.loads(row[0]) for row in connection.execute("SELECT document FROM images ORDER BY vector_id")]
        if self.index.ntotal != len(self.rows) or len(self.rows) != self.metadata["images"]:
            raise ValueError("Index/catalog row mismatch")

    def search(self, query, *, limit=5, category=None, brand=None, max_price=None, attributes=None, prefer_low_price=False):
        # Exact small-catalog search: scan every vector so filtering cannot hide valid candidates.
        # For larger catalogs use sharded/filtered indexes, not a fixed unfiltered top-k.
        scores, ids = self.index.search(normalize(query), self.index.ntotal)
        grouped = {}
        for score, index_id in zip(scores[0], ids[0]):
            row = self.rows[int(index_id)]
            if category and row["category"] != category or brand and row["brand"] != brand:
                continue
            price = row["price_cny"]
            if max_price is not None and (price is None or price > max_price):
                continue
            if any(row["attributes"].get(k) != v for k, v in (attributes or {}).items()):
                continue
            if row["product_id"] in grouped:
                continue  # Max similarity over the product's gallery views.
            facts = {key: row[key] for key in ("product_id", "image_id", "title", "category", "brand", "price_cny", "attributes", "source")}
            grouped[row["product_id"]] = {**facts, "similarity": float(score), "preference_bonus": 0.0,
                                            "image_url": f"/api/v1/visual/images/{row['image_id']}"}
        results = list(grouped.values())
        prices = [r["price_cny"] for r in results if r["price_cny"] is not None]
        if prefer_low_price and prices and max(prices) > min(prices):
            for row in results:
                if row["price_cny"] is not None:
                    row["preference_bonus"] = .03 * (max(prices) - row["price_cny"]) / (max(prices) - min(prices))
        for row in results:
            row["ranking_score"] = row["similarity"] + row["preference_bonus"]
            row["evidence"] = {"snapshot": self.metadata["manifest_sha256"], "source": row["source"],
                               "matched_attributes": attributes or {}, "budget_checked": max_price is not None,
                               "applied_filters": {"category": category, "brand": brand, "max_price": max_price}}
        return sorted(results, key=lambda row: (-row["ranking_score"], row["product_id"]))[:limit]
