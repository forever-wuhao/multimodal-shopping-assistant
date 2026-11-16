"""Run from backend: python -m app.visual.cli --help."""
import argparse
import json
from pathlib import Path

from .data import load_manifest, split_identities


def main():
    parser = argparse.ArgumentParser(description="Local product retrieval data, training and evaluation")
    parser.add_argument("command", choices=["validate", "split", "train", "index", "evaluate", "compare"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--products-per-batch", type=int, default=8)
    args = parser.parse_args()
    if args.command != "validate" and not args.output:
        parser.error("--output is required")
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new path to preserve provenance")
    if args.epochs < 1 or args.products_per_batch < 2:
        parser.error("epochs >=1 and products-per-batch >=2 are required")
    if args.command == "compare" and not args.adapter:
        parser.error("compare requires --adapter")
    if args.command == "train" and args.adapter:
        parser.error("train starts from a base model; omit --adapter")
    rows = load_manifest(args.manifest, args.root, require_split=args.command not in ("split", "validate"))
    if args.command == "validate":
        print(json.dumps({"images": len(rows), "products": len({r.product_id for r in rows})}))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "split":
        rows = split_identities(rows, args.seed)
        args.output.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
        return
    from .encoder import ClipEncoder
    encoder = ClipEncoder(args.model, args.pretrained, args.device, args.adapter if args.command not in ("train", "compare") else None)
    if args.command == "index":
        from .index import build_snapshot
        result = build_snapshot(rows, args.root, args.output, encoder, args.adapter)
    elif args.command == "train":
        from .training import train_adapter
        result = train_adapter(rows, args.root, encoder, args.output, epochs=args.epochs, seed=args.seed, products_per_batch=args.products_per_batch)
    else:
        from .training import evaluate, compare
        result = compare(rows, args.root, encoder, args.adapter) if args.command == "compare" else evaluate(rows, args.root, encoder)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
