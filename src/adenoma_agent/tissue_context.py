"""Compatibility imports for the Mucosa Extractor v1 tissue-context API.

New code should import :mod:`adenoma_agent.mucosa_extractor` directly.  This
module remains because the PathPrism server and historical experiment scripts
import the probability mapping helpers from here.
"""

from adenoma_agent.mucosa_extractor import (  # noqa: F401
    CRC100K_CONTEXT_MAP,
    CRC100K_LABELS,
    TISSUE_CONTEXT_COLORS,
    TISSUE_CONTEXT_LABELS,
    MucosaExtractorConfig,
    mucosa_score,
    normalized_entropy,
    run_mucosa_extractor,
    tissue_context_from_probabilities as _crc100k_tissue_context_from_probabilities,
    tissue_context_label,
    validate_crc100k_probabilities,
)


def tissue_context_from_probabilities(probabilities, model_name="uni_prismnet", strict=True):
    """Map model probabilities into the v1 seven-channel vocabulary.

    DIgePath support is retained only for historical experiment readers. The
    canonical Mucosa Extractor always uses CRC100K probabilities from PathPrism.
    """

    if str(model_name or "").strip().lower() != "digepath":
        return _crc100k_tissue_context_from_probabilities(probabilities, model_name=model_name, strict=strict)
    aliases = {
        "ADI": "adipose",
        "BACK": "background",
        "DEB": "debris",
        "LYM": "lymphocytes",
        "MUC": "mucus",
        "MUS": "smooth_muscle",
        "NORM": "normal_colon_mucosa",
        "STR": "stroma",
        "TUM": "tumor_epithelium",
    }
    translated = {label: float((probabilities or {}).get(source, 0.0) or 0.0) for label, source in aliases.items()}
    return _crc100k_tissue_context_from_probabilities(translated, model_name="uni_prismnet", strict=strict)


def build_tissue_context_maps(
    artifact_dir,
    output_dir=None,
    crop_id="20x_512",
    source_model="uni_prismnet",
    threshold=0.30,
    mask_downsample=32.0,
    min_tissue_coverage=0.0,
    mask_opacity=0.20,
    context_opacity=0.50,
    grid_opacity=0.20,
):
    """Run the v1 extractor from a saved artifact bundle.

    Visualization opacity arguments are accepted for source compatibility but
    v1 uses a single standardized overlay style.
    """

    del crop_id, mask_opacity, context_opacity, grid_opacity
    config = MucosaExtractorConfig(
        source_model=source_model,
        mucosa_threshold=float(threshold),
        mask_downsample=float(mask_downsample),
        min_tissue_coverage=float(min_tissue_coverage),
    )
    return run_mucosa_extractor(
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        config=config,
    )
