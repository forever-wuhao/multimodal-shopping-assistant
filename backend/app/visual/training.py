"""Supervised contrastive adaptation and identity-disjoint retrieval evaluation."""
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np

from .data import eval_partition, image_path, manifest_digest, read_image
from .encoder import normalize, load_adapter


def encode_rows(encoder, rows, root, batch_size=16):
    return np.concatenate([encoder.images([read_image(image_path(root, r.image)) for r in rows[i:i + batch_size]])
                           for i in range(0, len(rows), batch_size)])


def supervised_contrastive_loss(features, labels, temperature=.07):
    import torch
    import torch.nn.functional as F
    z = F.normalize(features, dim=-1)
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    if not positives.any(dim=1).all():
        raise ValueError("Every anchor must have another view of the same product")
    logits = (z @ z.T / temperature).masked_fill(diagonal, -1e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -((log_probability * positives).sum(1) / positives.sum(1)).mean()


def retrieval_metrics(gallery_features, gallery_ids, query_features, query_ids, index=None):
    import faiss
    if index is None:
        index = faiss.IndexFlatIP(gallery_features.shape[1])
        index.add(normalize(gallery_features))
    _, neighbors = index.search(normalize(query_features), len(gallery_ids))
    ranked = [list(dict.fromkeys(gallery_ids[i] for i in indices))[:5] for indices in neighbors]
    hits1 = [rank[0] == pid for rank, pid in zip(ranked, query_ids)]
    hits5 = [pid in rank for rank, pid in zip(ranked, query_ids)]
    macro = [np.mean([hit for hit, q in zip(hits1, query_ids) if q == pid]) for pid in set(query_ids)]
    return {"hit_at_1": float(np.mean(hits1)), "hit_at_5": float(np.mean(hits5)),
            "macro_hit_at_1": float(np.mean(macro))}, ranked


def train_adapter(rows, root, encoder, output, *, epochs=30, products_per_batch=8, rank=64, learning_rate=.001, seed=42):
    import torch
    from torch import nn
    import torch.nn.functional as F
    if encoder.adapter:
        raise ValueError("Training starts from a base encoder, not an existing adapter")
    output = Path(output)
    if output.exists():
        raise ValueError("Adapter output already exists")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train = [r for r in rows if r.split == "train"]
    groups = {}
    for i, row in enumerate(train):
        groups.setdefault(row.product_id, []).append(i)
    if len(groups) < 2 or any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("Training requires >=2 products, each with >=2 distinct views")
    val_gallery, val_query = eval_partition(rows, "val")
    # Test images are never encoded or consulted during training/model selection.
    features = encode_rows(encoder, train, root)
    texts = np.concatenate([encoder.texts([r.text for r in train[i:i+16]]) for i in range(0, len(train), 16)])
    vg, vq = encode_rows(encoder, val_gallery, root), encode_rows(encoder, val_query, root)
    ids = sorted(groups)
    centroids = normalize(np.stack([features[groups[pid]].mean(0) for pid in ids]))
    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = {pid: ids[int(similarity[i].argmax())] for i, pid in enumerate(ids)}
    hard_pairs = [{"anchor": pid, "negative": nearest[pid]} for pid in ids]
    dim = features.shape[1]
    rank = min(rank, dim)
    head = nn.Sequential(nn.Linear(dim, rank, bias=False), nn.ReLU(), nn.Linear(rank, dim, bias=False)).to(encoder.device)
    nn.init.zeros_(head[2].weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.01)
    x = torch.tensor(features, device=encoder.device)
    t = torch.tensor(texts, device=encoder.device)
    label_ids = {pid: i for i, pid in enumerate(ids)}
    labels = torch.tensor([label_ids[r.product_id] for r in train], device=encoder.device)
    batch_products = min(max(2, products_per_batch), len(ids))
    history, best, best_epoch, best_weights = [], -1., 0, None
    for epoch in range(1, epochs + 1):
        head.train()
        losses = []
        anchors = ids.copy()
        rng.shuffle(anchors)
        for step in range(math.ceil(len(ids) / batch_products)):
            anchor = anchors[(step * batch_products) % len(ids)]
            # Always include the visually nearest *different training identity*.
            chosen = [anchor, nearest[anchor]]
            chosen += rng.sample([pid for pid in ids if pid not in chosen], batch_products - 2)
            indices = [idx for pid in chosen for idx in rng.sample(groups[pid], 2)]
            image_z = F.normalize(x[indices] + head(x[indices]), dim=-1)
            text_z = F.normalize(t[indices] + head(t[indices]), dim=-1)
            loss = supervised_contrastive_loss(image_z, labels[indices])
            # Preserve cross-modal compatibility while fitting image-to-image identity.
            loss = loss + .2 * supervised_contrastive_loss(torch.cat([image_z, text_z]), labels[indices].repeat(2))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.inference_mode():
            def project(array):
                value = torch.tensor(array, device=encoder.device)
                return F.normalize(value + head(value), dim=-1).cpu().numpy()
            metrics, _ = retrieval_metrics(project(vg), [r.product_id for r in val_gallery], project(vq), [r.product_id for r in val_query])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": metrics})
        if metrics["macro_hit_at_1"] > best:
            best, best_epoch = metrics["macro_hit_at_1"], epoch
            best_weights = head[0].weight.detach().cpu().numpy().copy(), head[2].weight.detach().cpu().numpy().copy()
        print(json.dumps(history[-1]), flush=True)
    metadata = {"schema": 1, "method": "frozen_openclip_residual_adapter_supcon", "encoder": encoder.config,
                "manifest_sha256": manifest_digest(rows), "seed": seed, "epochs": epochs,
                "rank": rank, "learning_rate": learning_rate, "products_per_batch": batch_products,
                "selected_epoch": best_epoch, "selection_metric": "validation_macro_hit_at_1",
                "train_product_ids": ids, "validation_product_ids": sorted({r.product_id for r in val_gallery + val_query}),
                "hard_negatives": hard_pairs, "history": history}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        np.savez(file, down=best_weights[0], up=best_weights[1], metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    return metadata


def evaluate(rows, root, encoder, split="test"):
    import torch
    gallery, queries = eval_partition(rows, split)
    if encoder.adapter_metadata:
        trained = set(encoder.adapter_metadata["train_product_ids"])
        selected = set(encoder.adapter_metadata["validation_product_ids"])
        tested = {r.product_id for r in gallery + queries}
        if tested & (trained | (selected if split == "test" else set())):
            raise ValueError("Evaluation identities overlap adapter training/model-selection identities")
    gf = encode_rows(encoder, gallery, root)
    import faiss
    index = faiss.IndexFlatIP(gf.shape[1])
    index.add(normalize(gf))
    # Warm-up excluded; measured per-query time includes preprocess, encode, FAISS + product dedup.
    encoder.images([read_image(image_path(root, queries[0].image))])
    latencies, predictions, failures = [], [], []
    for query in queries:
        image = read_image(image_path(root, query.image))
        started = time.perf_counter()
        qf = encoder.images([image])
        _, ranked = retrieval_metrics(gf, [r.product_id for r in gallery], qf, [query.product_id], index=index)
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(qf)
        if ranked[0][0] != query.product_id:
            failures.append({"image_id": query.image_id, "expected": query.product_id, "top5": ranked[0]})
    metrics, _ = retrieval_metrics(gf, [r.product_id for r in gallery], np.concatenate(predictions), [r.product_id for r in queries])
    return {**metrics, "mean_query_ms": float(np.mean(latencies)), "p95_query_ms": float(np.percentile(latencies, 95)),
            "query_images": len(queries), "gallery_images": len(gallery), "gallery_products": len({r.product_id for r in gallery}),
            "device": encoder.device, "device_name": torch.cuda.get_device_name(encoder.device) if encoder.device.startswith("cuda") else platform.processor(),
            "torch_version": torch.__version__, "encoder": encoder.identity(), "split": split,
            "manifest_sha256": manifest_digest(rows), "failures": failures,
            "protocol": "closed-set; unique product ranking; query/gallery different images; excludes disk decode/network/index build; batch=1; warmup=1",
            "warning": "Top5 is trivial for <=5 gallery products" if len({r.product_id for r in gallery}) <= 5 else None}


def compare(rows, root, encoder, adapter):
    baseline = evaluate(rows, root, encoder)
    metadata, down, up = load_adapter(adapter)
    if metadata["encoder"] != encoder.config:
        raise ValueError("Adapter backbone mismatch")
    import hashlib
    encoder.adapter = down, up
    encoder.adapter_metadata = metadata
    encoder.adapter_sha = hashlib.sha256(Path(adapter).read_bytes()).hexdigest()
    adapted = evaluate(rows, root, encoder)
    return {"baseline": baseline, "adapted": adapted,
            "delta_hit_at_1_percentage_points": 100 * (adapted["hit_at_1"] - baseline["hit_at_1"])}
