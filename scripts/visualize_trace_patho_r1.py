#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from adenoma_agent.agents.trace import TraceAgent
from adenoma_agent.multimodal import _build_trace_output_from_text, _build_trace_patho_r1_prompt
from adenoma_agent.utils import ensure_dir, run_command, write_json, write_text


DEFAULT_TRACE_PROMPT = (
    "You are the Trace Agent in a hierarchical pathology workflow. "
    "Given a thumbnail image of a colorectal whole-slide image and a list of candidate proposal boxes, "
    "identify which proposals most likely contain serrated suspicious mucosa. "
    "Focus only on mucosa and serrated screening. "
    "Do not assess abnormal crypt or dysplasia. "
    "Return JSON only with selected proposal cluster_id values and fields l, s, d, review_stage, desc, and evidence."
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Patho-R1 on a thumbnail for trace-region selection and save an annotated visualization."
    )
    parser.add_argument("--thumbnail-path", required=True, help="Path to the thumbnail image.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for visualization and JSON outputs. Defaults to artifacts/trace_visualizations/<thumbnail_stem>.",
    )
    parser.add_argument(
        "--patho-r1-python",
        default="/data1/yuexin/.conda/envs/patho-r1/bin/python",
        help="Python interpreter for the local Patho-R1 environment.",
    )
    parser.add_argument(
        "--patho-r1-runner",
        default="/data1/yuexin/patho-r1-3b/run_patho_r1.py",
        help="Path to the local Patho-R1 runner script.",
    )
    parser.add_argument(
        "--model-id",
        default="WenchuanZhang/Patho-R1-3B",
        help="Hugging Face model id passed through to Patho-R1.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint or mirror used when loading the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens requested from Patho-R1.",
    )
    parser.add_argument(
        "--max-trace-candidates",
        type=int,
        default=5,
        help="Maximum selected trace clusters to keep for the visualization.",
    )
    parser.add_argument(
        "--cluster-grid-size",
        type=int,
        default=16,
        help="Grid size used when building thumbnail proposals.",
    )
    parser.add_argument(
        "--min-cell-tissue-fraction",
        type=float,
        default=0.08,
        help="Minimum per-cell tissue fraction for proposal generation.",
    )
    parser.add_argument(
        "--min-cluster-area-fraction",
        type=float,
        default=0.01,
        help="Minimum connected-component area fraction for proposal generation.",
    )
    parser.add_argument(
        "--trace-prompt",
        default=DEFAULT_TRACE_PROMPT,
        help="Prompt template for Patho-R1 trace selection.",
    )
    return parser


def build_bundle(args):
    return {
        "runtime": {
            "trace": {
                "cluster_grid_size": args.cluster_grid_size,
                "min_cell_tissue_fraction": args.min_cell_tissue_fraction,
                "min_cluster_area_fraction": args.min_cluster_area_fraction,
                "labels": [
                    "serrated_suspicious_mucosa",
                    "non_serrated_mucosa",
                    "background",
                    "artifact",
                ],
            }
        },
        "budget": {
            "max_trace_candidates": args.max_trace_candidates,
        },
    }


def build_proposals(bundle, thumbnail_path):
    image = Image.open(thumbnail_path)
    width, height = image.size
    trace_agent = TraceAgent(bundle, selector_adapter=None, backend_chain=None)
    thumbnail_meta = {
        "thumbnail_size": [width, height],
        "slide_dimensions_level0": [width, height],
    }
    return trace_agent._build_proposals(thumbnail_path, thumbnail_meta, route_c_boxes=[])


def invoke_patho_r1(args, prompt, thumbnail_path):
    command = [
        args.patho_r1_python,
        args.patho_r1_runner,
        "--image",
        str(thumbnail_path),
        "--prompt",
        prompt,
        "--model-id",
        args.model_id,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    env_overrides = {
        "HF_ENDPOINT": args.hf_endpoint,
        "HUGGINGFACE_HUB_ENDPOINT": args.hf_endpoint,
    }
    result = run_command(command, timeout=1800, env_overrides=env_overrides)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or "Patho-R1 trace command failed")
    text = (result["stdout"] or "").strip()
    if not text:
        raise RuntimeError("Patho-R1 trace produced empty output")
    return text, result


def enrich_clusters(trace_output, proposals):
    proposal_lookup = {proposal["cluster_id"]: proposal for proposal in proposals}
    clusters = []
    for cluster in trace_output["clusters"]:
        proposal = proposal_lookup.get(cluster["cluster_id"])
        if proposal is None:
            continue
        clusters.append(
            {
                **cluster,
                "cluster_bbox_thumb": proposal["cluster_bbox_thumb"],
                "metadata": {
                    **proposal.get("metadata", {}),
                    **cluster.get("metadata", {}),
                },
            }
        )
    return clusters


def draw_clusters(thumbnail_path, clusters, output_path):
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [
        (255, 64, 64),
        (255, 140, 0),
        (255, 200, 0),
        (50, 160, 255),
        (120, 220, 120),
    ]
    for index, cluster in enumerate(clusters):
        bbox = cluster["cluster_bbox_thumb"]
        color = colors[index % len(colors)]
        draw.rectangle([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], outline=color, width=8)
        tag_x1 = bbox["x1"]
        tag_y1 = max(0, bbox["y1"] - 42)
        tag_x2 = min(image.size[0], tag_x1 + 250)
        tag_y2 = min(image.size[1], tag_y1 + 42)
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=color)
        draw.text((tag_x1 + 8, tag_y1 + 10), "{0} s={1}".format(cluster["cluster_id"], cluster["s"]), fill=(0, 0, 0))
    image.save(output_path)


def main():
    parser = build_parser()
    args = parser.parse_args()

    thumbnail_path = Path(args.thumbnail_path).resolve()
    if not thumbnail_path.exists():
        raise SystemExit("Thumbnail not found: {0}".format(thumbnail_path))

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "artifacts" / "trace_visualizations" / thumbnail_path.stem
    )
    ensure_dir(output_dir)

    bundle = build_bundle(args)
    proposals = build_proposals(bundle, str(thumbnail_path))
    request = {
        "stage": "trace",
        "images": [str(thumbnail_path)],
        "prompt": {
            "question": args.trace_prompt,
            "task": "mucosa_serrated_abnormal_crypt_trace_annotation",
        },
        "metadata": {
            "case_id": thumbnail_path.stem,
            "thumbnail_meta": {
                "thumbnail_size": list(Image.open(thumbnail_path).size),
                "slide_dimensions_level0": list(Image.open(thumbnail_path).size),
            },
            "proposals": proposals,
            "selector_mode": "thumbnail_only",
        },
    }
    prompt = _build_trace_patho_r1_prompt(request, bundle)
    raw_text, command_result = invoke_patho_r1(args, prompt, thumbnail_path)
    trace_output = _build_trace_output_from_text(raw_text, request, bundle)
    clusters = enrich_clusters(trace_output, proposals)[: args.max_trace_candidates]

    visualization_path = output_dir / "{0}_trace_patho_r1.png".format(thumbnail_path.stem)
    clusters_json_path = output_dir / "{0}_trace_clusters.json".format(thumbnail_path.stem)
    prompt_path = output_dir / "{0}_trace_prompt.txt".format(thumbnail_path.stem)
    raw_response_path = output_dir / "{0}_trace_raw_response.txt".format(thumbnail_path.stem)
    summary_path = output_dir / "{0}_trace_summary.txt".format(thumbnail_path.stem)

    draw_clusters(thumbnail_path, clusters, visualization_path)
    write_json(
        clusters_json_path,
        {
            "thumbnail_path": str(thumbnail_path),
            "selected_clusters": clusters,
            "proposal_count": len(proposals),
            "backend_command": command_result["command"],
        },
    )
    write_text(prompt_path, prompt + "\n")
    write_text(raw_response_path, raw_text + "\n")

    summary_lines = [
        "thumbnail: {0}".format(thumbnail_path),
        "proposal_count: {0}".format(len(proposals)),
        "selected_cluster_count: {0}".format(len(clusters)),
        "visualization: {0}".format(visualization_path),
        "clusters_json: {0}".format(clusters_json_path),
    ]
    for cluster in clusters:
        bbox = cluster["cluster_bbox_thumb"]
        summary_lines.append(
            "{cluster_id}: bbox=({x1},{y1},{x2},{y2}), label={label}, priority={priority}, desc={desc}".format(
                cluster_id=cluster["cluster_id"],
                x1=bbox["x1"],
                y1=bbox["y1"],
                x2=bbox["x2"],
                y2=bbox["y2"],
                label=cluster["l"],
                priority=cluster["s"],
                desc=cluster["desc"],
            )
        )
    write_text(summary_path, "\n".join(summary_lines) + "\n")

    print("visualization={0}".format(visualization_path))
    print("clusters_json={0}".format(clusters_json_path))
    print("prompt={0}".format(prompt_path))
    print("raw_response={0}".format(raw_response_path))
    print("summary={0}".format(summary_path))


if __name__ == "__main__":
    main()
