#!/usr/bin/env python3
"""
Patho-R1 Inference Script for Adenoma Selection (Route C)

This script implements Phase 2C: Use Patho-R1 to perform intelligent selection on 1X thumbnails.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline
import openslide


def generate_thumbnail(slide_path: str, output_path: str, level: int = 0) -> None:
    """
    Generate thumbnail from WSI at specified level.

    Args:
        slide_path: Path to .svs file
        output_path: Path to save thumbnail
        level: Pyramid level (0 = highest resolution)
    """
    print(f"Generating thumbnail from {slide_path} at level {level}")
    sys.stdout.flush()
    slide = openslide.OpenSlide(slide_path)
    print("Opened slide")
    sys.stdout.flush()
    thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
    print("Thumbnail created")
    sys.stdout.flush()
    # Save as JPEG for faster writing
    output_path_jpg = output_path.replace('.png', '.jpg')
    thumbnail.save(output_path_jpg, 'JPEG', quality=85)
    slide.close()
    print(f"Thumbnail saved to {output_path_jpg}")
    sys.stdout.flush()
    return output_path_jpg


def run_patho_r1_inference(image_path: str, model_name: str = "WenchuanZhang/Patho-R1-3B", hf_token: str = None) -> str:
    """
    Run Patho-R1 inference on an image.

    Args:
        image_path: Path to input image
        model_name: HuggingFace model name
        hf_token: Hugging Face token for gated repo access

    Returns:
        Model response text
    """
    try:
        print(f"Loading model: {model_name}")
        sys.stdout.flush()
        # Load pipeline with optional token
        pipe_kwargs = {"model": model_name}
        if hf_token:
            pipe_kwargs["token"] = hf_token
        pipe = pipeline("image-text-to-text", **pipe_kwargs)
        print("Pipeline loaded successfully")
        sys.stdout.flush()

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Identify and outline the regions containing adenoma tissue in this pathology image thumbnail. Provide coordinates or bounding boxes."}
                ]
            }
        ]

        print("Running inference...")
        sys.stdout.flush()
        # Run inference in chat format with image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Identify and outline the regions containing adenoma tissue in this pathology image thumbnail. Provide coordinates or bounding boxes."}
                ]
            }
        ]
        output = pipe(text=messages)
        response = output[0].get('generated_text', str(output[0])) if isinstance(output, list) and output else str(output)
        print("Inference completed")
        sys.stdout.flush()
        return response

    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        print(error_msg)
        sys.stdout.flush()
        return error_msg


def main():
    parser = argparse.ArgumentParser(description="Patho-R1 inference for adenoma selection")
    parser.add_argument("--slide-path", required=True, help="Path to WSI file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-name", default="WenchuanZhang/Patho-R1-3B",
                       choices=["WenchuanZhang/Patho-R1-3B", "WenchuanZhang/Patho-R1-7B"],
                       help="Patho-R1 model to use")
    parser.add_argument("--thumbnail-level", type=int, default=1,
                       help="Thumbnail level (0=highest resolution)")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Hugging Face token for gated model access")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate slide ID from filename
    slide_id = Path(args.slide_path).stem

    # Generate thumbnail
    print("Starting thumbnail generation")
    sys.stdout.flush()
    thumbnail_path = output_dir / f"{slide_id}_thumbnail.png"
    actual_thumbnail_path = generate_thumbnail(args.slide_path, str(thumbnail_path), args.thumbnail_level)

    # Run Patho-R1 inference
    print(f"Running Patho-R1 inference on {actual_thumbnail_path}")
    sys.stdout.flush()
    response = run_patho_r1_inference(actual_thumbnail_path, args.model_name, args.hf_token)

    # Save response
    response_path = output_dir / f"{slide_id}_patho_r1_response.txt"
    with open(response_path, 'w') as f:
        f.write(response)

    print(f"Response saved to {response_path}")
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()