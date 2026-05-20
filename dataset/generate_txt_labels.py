import json
import os
import argparse
from PIL import Image

def main(json_path, images_root, labels_out):
    os.makedirs(labels_out, exist_ok=True)

    with open(json_path) as f:
        annotations = json.load(f)

    for _, entry in annotations.items():
        filename = entry["filename"]
        img_path = os.path.join(images_root, filename)
        if not os.path.isfile(img_path):
            continue

        img = Image.open(img_path)
        W, H = img.size

        lines = []
        for region in entry.get("regions", []):
            shape = region.get("shape_attributes", {})
            x, y, w, h = shape.get("x"), shape.get("y"), shape.get("width"), shape.get("height")
            if None in (x, y, w, h):
                continue

            # Convert to YOLO: center-based, normalized
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")  # class 0 = pigeon

        stem = os.path.splitext(filename)[0]
        with open(os.path.join(labels_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))  # empty file = no object (YOLO handles this fine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="dataset/labels.json")
    parser.add_argument("--images_root", default="dataset/images")
    parser.add_argument("--labels_out", default="dataset/labels")
    args = parser.parse_args()
    main(args.json_path, args.images_root, args.labels_out)
