"""Real OpenCLIP + adapter + FAISS + HTTP smoke, using explicitly synthetic images.

Run with --output data/smoke-v1. Does NOT measure real-world retrieval quality.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from app.visual.data import Sample, load_manifest, split_identities
from app.visual.encoder import ClipEncoder
from app.visual.index import build_snapshot
from app.visual.training import train_adapter, compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "pink", "brown"]
    rows = []
    for pid, color in enumerate(colors):
        for view in range(3):
            image_id = f"fixture_{pid}_{view}"
            image = Image.new("RGB", (224, 224), (210 + view * 10, 210, 210))
            draw = ImageDraw.Draw(image)
            draw.rectangle((30 + view * 5, 40, 170, 180 - view * 5), fill=color)
            draw.ellipse((60, 60 + pid * 4, 80 + pid * 3, 100), fill="white")
            image.save(root / f"{image_id}.png")
            rows.append(Sample(image_id=image_id, product_id=f"fixture_{pid}", image=f"{image_id}.png",
                title=f"合成流程测试 {color}", text=f"a {color} square with a white mark", category="合成测试",
                price_cny=pid * 10., attributes={"color": color}, source="合成流程测试图片，非真实商品，无业务准确率含义"))
    rows = split_identities(rows)
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(row.model_dump_json() for row in rows), encoding="utf-8")
    rows = load_manifest(manifest, root)
    encoder = ClipEncoder()
    adapter = root / "fixture-adapter.npz"
    train_adapter(rows, root, encoder, adapter, epochs=2, products_per_batch=4)
    comparison = compare(rows, root, encoder, adapter)
    comparison["WARNING"] = "Synthetic integration fixture only; NOT real product accuracy"
    (root / "fixture-comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    snapshot = root / "index"
    build_snapshot(rows, root, snapshot, encoder, adapter)
    # The startup path reconstructs the same encoder and verifies the copied weights.
    del encoder
    import gc
    gc.collect()
    from fastapi.testclient import TestClient
    from app.main import app
    from app.config import settings
    old = settings.visual_index_path
    settings.visual_index_path = str(snapshot)
    try:
        with TestClient(app) as client:
            assert client.get("/api/v1/visual/status").json()["ready"]
            query = next(r for r in rows if r.split == "test" and r.role == "query")
            response = client.post("/api/v1/visual/search", files={"image": ("test.png", (root / query.image).read_bytes())},
                                   data={"max_price": "50", "prefer_low_price": "true"})
            assert response.status_code == 200, response.text
            assert response.json()["results"]
            assert all(row["price_cny"] <= 50 for row in response.json()["results"])
            assert response.json()["encoder"]["adapter_sha256"]
    finally:
        settings.visual_index_path = old
    print(json.dumps({"result": "passed", "checks": ["real_openclip", "adapter_training", "held_out_comparison", "faiss_sqlite", "http_image_search"],
                      "synthetic_fixture_only": True}))


if __name__ == "__main__":
    main()
