"""Deterministic fixtures verify contracts, not real-world model accuracy."""
import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

from app.visual.data import Sample, load_manifest, split_identities, manifest_digest
from app.visual.encoder import normalize, load_adapter
from app.visual.index import ProductIndex, build_snapshot
from app.visual.training import train_adapter, evaluate, supervised_contrastive_loss


class FixtureEncoder:
    """Test-only embeddings: never used in the application or reported as CLIP."""
    config = {"model": "fixture", "pretrained": "none"}
    device = "cpu"
    adapter = adapter_metadata = adapter_sha = None

    def images(self, images):
        values = np.array([np.array(image).mean((0, 1)).tolist() + [1.] for image in images], dtype="float32")
        return self.project(values)

    def texts(self, texts):
        return self.project(np.array([[len(text), 2., 3., 1.] for text in texts], dtype="float32"))

    def project(self, values):
        values = normalize(values)
        if self.adapter:
            down, up = self.adapter
            values = values + np.maximum(values @ down.T, 0) @ up.T
        return normalize(values)

    def identity(self):
        return {**self.config, "adapter_sha256": self.adapter_sha}


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for product in range(8):
        for view in range(3):
            filename = f"p{product}_v{view}.png"
            image = Image.new("RGB", (24, 24), (20 + product * 25, 40 + view * 20, 100))
            ImageDraw.Draw(image).rectangle((view, 2, 5 + product, 10), fill="white")
            image.save(tmp_path / filename)
            rows.append(Sample(image_id=f"p{product}_v{view}", product_id=f"p{product}", image=filename,
                title=f"Fixture {product}", text=f"Test object {product}, view {view}", category="fixture",
                price_cny=None if product == 0 else product * 100., attributes={"color": "blue" if product % 2 else "red"},
                source="Synthetic test fixture, not a real product"))
    rows = split_identities(rows)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(row.model_dump_json() for row in rows), encoding="utf-8")
    return rows, manifest, tmp_path


def test_split_and_manifest_reject_identity_leakage(dataset):
    rows, manifest, root = dataset
    validated = load_manifest(manifest, root)
    assert len(validated) == 24
    for pid in {r.product_id for r in rows}:
        assert len({r.split for r in rows if r.product_id == pid}) == 1
    changed = rows.copy()
    changed[0] = rows[0].model_copy(update={"split": "train" if rows[0].split == "test" else "test"})
    manifest.write_text("\n".join(r.model_dump_json() for r in changed), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(manifest, root)


def test_reject_pixel_duplicates_and_path_escape(dataset):
    rows, manifest, root = dataset
    original = (root / rows[1].image).read_bytes()
    (root / rows[1].image).write_bytes((root / rows[0].image).read_bytes())
    with pytest.raises(ValueError, match="Duplicate image pixels"):
        load_manifest(manifest, root)
    (root / rows[1].image).write_bytes(original)
    rows[0] = rows[0].model_copy(update={"image": "../outside.png"})
    manifest.write_text("\n".join(r.model_dump_json() for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        load_manifest(manifest, root)


def test_index_filters_unknown_prices_deduplicates_and_checks_integrity(dataset):
    rows, _, root = dataset
    # Multiple gallery images for the same product must occupy only one result.
    gallery = [r.model_copy(update={"role": "gallery"}) for r in rows]
    output = root / "index"
    build_snapshot(gallery, root, output, FixtureEncoder())
    index = ProductIndex(output)
    results = index.search([[1, 1, 1, 1]], limit=20)
    assert len(results) == 8 and len({r["product_id"] for r in results}) == 8
    filtered = index.search([[1, 1, 1, 1]], max_price=300, attributes={"color": "blue"}, prefer_low_price=True)
    assert {r["product_id"] for r in filtered} == {"p1", "p3"}
    assert all(r["evidence"]["budget_checked"] for r in filtered)
    assert index.search([[1, 1, 1, 1]], max_price=0) == []
    with pytest.raises(ValueError, match="already exists"):
        build_snapshot(gallery, root, output, FixtureEncoder())
    with (output / "vectors.faiss").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        ProductIndex(output)


def test_train_and_evaluate_use_disjoint_identities(dataset):
    rows, _, root = dataset
    encoder = FixtureEncoder()
    adapter = root / "adapter.npz"
    result = train_adapter(rows, root, encoder, adapter, epochs=2, products_per_batch=2, rank=2)
    assert result["manifest_sha256"] == manifest_digest(rows)
    assert all(pair["anchor"] != pair["negative"] for pair in result["hard_negatives"])
    metadata, down, up = load_adapter(adapter)
    assert np.any(up != 0), "Optimizer must actually update adapter weights"
    encoder.adapter, encoder.adapter_metadata = (down, up), metadata
    metrics = evaluate(rows, root, encoder)
    assert 0 <= metrics["hit_at_1"] <= 1
    assert metrics["query_images"] == 4 and metrics["mean_query_ms"] > 0
    encoder.adapter_metadata["train_product_ids"].append(next(r.product_id for r in rows if r.split == "test"))
    with pytest.raises(ValueError, match="overlap"):
        evaluate(rows, root, encoder)


def test_loss_has_gradients_and_requires_positive_pairs():
    import torch
    features = torch.randn(4, 8, requires_grad=True)
    loss = supervised_contrastive_loss(features, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(loss) and features.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="another view"):
        supervised_contrastive_loss(features, torch.tensor([0, 1, 2, 3]))


def test_api_unavailable_and_input_validation():
    from app.main import app
    from app.config import settings
    previous = settings.visual_index_path
    settings.visual_index_path = ""
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/api/v1/visual/status").json()["ready"] is False
            assert client.post("/api/v1/visual/search", data={"text": "shoe"}).status_code == 503
            assert client.post("/api/v1/visual/search").status_code == 422
            assert client.post("/api/v1/visual/search", files={"image": ("bad.jpg", b"not an image")}).status_code == 422
            assert client.post("/api/v1/visual/search", data={"text": "shoe", "attributes": "[]"}).status_code == 422
            assert client.post("/api/v1/visual/search", data={"text": "shoe", "max_price": "inf"}).status_code == 422
            assert client.get("/api/v1/knowledge/products").status_code == 200
            # Existing no-API shopping requirements remain operational.
            parsed = client.post("/api/v1/requirements/parse", json={"message": "推荐耳机，预算500元"})
            assert parsed.status_code == 200
            assert parsed.json()["hard_constraints"]
    finally:
        settings.visual_index_path = previous


def test_api_image_search_serves_evidence_and_thumbnails(dataset):
    import threading
    from app.main import app
    from app.routers.visual import VisualSearch
    rows, _, root = dataset
    output = root / "api-index"
    build_snapshot(rows, root, output, FixtureEncoder())
    service = VisualSearch.__new__(VisualSearch)
    service.index, service.encoder, service.lock = ProductIndex(output), FixtureEncoder(), threading.Lock()
    with TestClient(app) as client:
        app.state.visual_search = service
        photo = (root / rows[0].image).read_bytes()
        response = client.post("/api/v1/visual/search", files={"image": ("photo.png", photo)}, data={"max_price": "300"})
        assert response.status_code == 200
        result = response.json()
        assert result["candidate_only"] is True
        assert all(r["price_cny"] <= 300 for r in result["results"])
        assert client.get(result["results"][0]["image_url"]).headers["content-type"] == "image/jpeg"
        assert client.get("/api/v1/visual/images/missing").status_code == 404
        with service.lock:
            assert client.post("/api/v1/visual/search", data={"text": "test"}).status_code == 429
