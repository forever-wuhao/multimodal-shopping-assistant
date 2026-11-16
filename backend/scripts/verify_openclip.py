"""Optional real-weight smoke check. No retrieval accuracy claim or dataset required."""
import argparse
import json
from PIL import Image
import numpy as np
from app.visual.encoder import ClipEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    encoder = ClipEncoder(device=args.device)
    image = Image.new("RGB", (256, 256), "red")
    vectors = encoder.images([image, image])
    text = encoder.texts(["a red square"])
    assert vectors.shape[0] == 2 and text.shape == vectors[:1].shape
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert np.allclose(vectors[0], vectors[1], atol=1e-5)
    print(json.dumps({"check": "real_pretrained_forward_pass", "encoder": encoder.identity(),
                      "dimension": vectors.shape[1], "device": args.device, "semantic_accuracy_evaluated": False}))


if __name__ == "__main__":
    main()
