#!/usr/bin/env python3
import argparse

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a pathology question about a single patch image using Patho-R1."
    )
    parser.add_argument("--image", required=True, help="Path to a PNG/JPG pathology patch")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument(
        "--model-id",
        default="WenchuanZhang/Patho-R1-3B",
        help="Hugging Face model id or local model path",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for model downloads",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a pathology expert assisting a preprocessing workflow. "
            "Briefly describe tissue quality, likely histologic content, and whether the patch "
            "looks diagnostically useful. Use the format: <think>...</think><answer>...</answer>"
        ),
        help="System prompt passed to the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=args.cache_dir,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)

    messages = [
        {"role": "system", "content": args.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.question},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
