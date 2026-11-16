"""A portable, provenance-aware image manifest with identity leakage checks."""
import hashlib
import json
import random
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    image_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    product_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    image: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=1000)
    category: str = Field(min_length=1, max_length=100)
    brand: str = Field(default="", max_length=100)
    price_cny: float | None = Field(default=None, ge=0)
    attributes: dict[str, str] = Field(default_factory=dict)
    source: str = Field(min_length=1, max_length=1000)
    split: Literal["unassigned", "train", "val", "test"] = "unassigned"
    role: Literal["gallery", "query"] = "gallery"


def image_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Image paths must stay inside the dataset directory")
    if not path.is_file():
        raise ValueError(f"Image missing: {relative}")
    return path


def read_image(path):
    with Image.open(path) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds 20 megapixels")
        return ImageOps.exif_transpose(image).convert("RGB")


def load_manifest(path: Path, root: Path, require_split=True) -> list[Sample]:
    rows = [Sample.model_validate_json(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or len(rows) > 50_000:
        raise ValueError("Manifest must contain 1..50000 images")
    image_ids, pixels, products = set(), {}, {}
    for row in rows:
        if row.image_id in image_ids:
            raise ValueError(f"Duplicate image_id: {row.image_id}")
        image_ids.add(row.image_id)
        image = read_image(image_path(root, row.image))
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in pixels:
            raise ValueError(f"Duplicate image pixels: {pixels[digest]} / {row.image_id}")
        pixels[digest] = row.image_id
        if require_split and row.split == "unassigned":
            raise ValueError("Run split before training, evaluation or indexing")
        # Text can describe a particular view. Product facts must agree across views.
        facts = row.model_dump(exclude={"image_id", "image", "text", "role"})
        if row.product_id in products and products[row.product_id] != facts:
            raise ValueError(f"Conflicting product facts or split leakage: {row.product_id}")
        products[row.product_id] = facts
    return rows


def split_identities(rows: list[Sample], seed=42) -> list[Sample]:
    groups = {}
    for row in rows:
        groups.setdefault(row.product_id, []).append(row)
    if len(groups) < 6 or any(len(group) < 2 for group in groups.values()):
        raise ValueError("Need >=6 products and >=2 distinct views per product")
    rng = random.Random(seed)
    ids = sorted(groups)
    rng.shuffle(ids)
    n_eval = max(2, round(len(ids) * .15))
    result = []
    for i, pid in enumerate(ids):
        split = "test" if i < n_eval else "val" if i < 2 * n_eval else "train"
        group = sorted(groups[pid], key=lambda row: row.image_id)
        rng.shuffle(group)
        for j, row in enumerate(group):
            result.append(row.model_copy(update={"split": split, "role": "gallery" if j == 0 else "query"}))
    return result


def manifest_digest(rows):
    payload = [row.model_dump() for row in sorted(rows, key=lambda item: item.image_id)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def eval_partition(rows, split):
    gallery = [r for r in rows if r.split == split and r.role == "gallery"]
    queries = [r for r in rows if r.split == split and r.role == "query"]
    if len({r.product_id for r in gallery}) < 2 or not queries:
        raise ValueError(f"{split} needs >=2 gallery products and separate query views")
    if not {r.product_id for r in queries} <= {r.product_id for r in gallery}:
        raise ValueError("Every query identity needs a gallery view")
    return gallery, queries
