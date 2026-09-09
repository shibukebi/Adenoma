#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import open_clip
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank exported pathology patch images against preprocessing QC prompts using Patho-CLIP."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing exported patch images")
    parser.add_argument("--prompts-file", required=True, help="Text file with one prompt per line")
    parser.add_argument("--weights-path", required=True, help="Path to local Patho-CLIP weight file (.pt)")
    parser.add_argument("--output-csv", required=True, help="Where to write the ranking results")
    parser.add_argument(
        "--model-name",
        default="ViT-L-14",
        help="open_clip model name corresponding to the provided weights",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def collect_images(image_dir: Path) -> list[Path]:
    images = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    if not images:
        raise ValueError(f"No image patches found in {image_dir}")
    return images


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    prompts_file = Path(args.prompts_file)
    weights_path = Path(args.weights_path)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_file)
    images = collect_images(image_dir)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Patho-CLIP weights not found at {weights_path}. Download the model first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=str(weights_path)
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)
    model.eval()

    text_tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "top_prompt", "top_score", "scores_json"])

        for image_path in images:
            image_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode():
                image_embedding = model.encode_image(image_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                scores = (image_embedding @ text_embeddings.T).squeeze(0).cpu().tolist()

            ranked = sorted(zip(prompts, scores), key=lambda item: item[1], reverse=True)
            top_prompt, top_score = ranked[0]
            writer.writerow([str(image_path), top_prompt, f"{top_score:.6f}", json.dumps(ranked, ensure_ascii=False)])

    print(f"Ranked {len(images)} images")
    print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
