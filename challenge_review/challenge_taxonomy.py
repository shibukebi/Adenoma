LESION_DIAGNOSES = ["HP", "SSL", "TSA", "TA", "TVA", "IP", "USA", "Other", "Ambiguous"]
HGD_STATUSES = ["Absent", "Present", "Not assessable", "Uncertain/conflicting"]
LABEL_ACTIONS = ["confirm_original", "correct_original", "remains_ambiguous"]
CHALLENGE_DISPOSITIONS = [
    "retain_challenge",
    "pending_label_adjudication",
    "exclude_label_error",
]
PRIMARY_CHALLENGES = [
    "HP vs SSL",
    "SSL vs TSA",
    "Serrated vs conventional adenoma",
    "TA vs TVA",
    "Adenoma vs inflammatory",
    "Focal HGD",
    "Borderline HGD",
    "Insufficient / fragmented specimen",
    "Other differential",
]
DIFFICULTY_MODIFIERS = [
    "focal_evidence",
    "subtle_morphology",
    "heterogeneous_lesion",
    "poor_orientation",
    "limited_sampling",
    "requires_high_magnification",
    "requires_multiple_regions",
    "technical_artifact",
    "competing_morphology",
]
NO_ROI_REASONS = [
    "global_architecture_required",
    "insufficient_tissue",
    "relevant_evidence_absent",
    "genuine_diagnostic_ambiguity",
    "technical_limitation",
    "other",
]
DIAGNOSTIC_ROLES = [
    "discriminative",
    "confirmatory",
    "contradictory",
    "mimic_confounder",
    "hgd_defining",
    "assessability_quality",
]
EVIDENCE_STRENGTHS = ["weak", "moderate", "strong", "decisive"]

GENERAL_EVIDENCE = [
    "surface_serration",
    "crypt_architecture",
    "cytologic_atypia",
    "inflammation",
    "tissue_quality_limitation",
    "other",
]
EVIDENCE_BY_CHALLENGE = {
    "HP vs SSL": [
        "basal_crypt_dilation", "horizontal_crypt_growth", "boot_l_shaped_crypt",
        "asymmetric_crypt_proliferation", "surface_serration",
    ],
    "SSL vs TSA": [
        "ectopic_crypt_formation", "eosinophilic_cytoplasm", "pencillate_nuclei",
        "slit_like_serration", "surface_serration",
    ],
    "Serrated vs conventional adenoma": [
        "serrated_architecture", "conventional_dysplasia", "crypt_elongation",
        "eosinophilic_cytoplasm", "villous_architecture",
    ],
    "TA vs TVA": ["tubular_architecture", "villous_architecture", "tubulovillous_architecture"],
    "Adenoma vs inflammatory": [
        "adenomatous_dysplasia", "regenerative_atypia", "lamina_propria_inflammation",
        "erosion_ulceration",
    ],
    "Focal HGD": [
        "complex_glandular_architecture", "cribriforming", "marked_cytologic_atypia",
        "loss_of_polarity", "luminal_necrosis",
    ],
    "Borderline HGD": [
        "complex_glandular_architecture", "cribriforming", "marked_cytologic_atypia",
        "loss_of_polarity", "luminal_necrosis",
    ],
    "Insufficient / fragmented specimen": [
        "fragmentation", "poor_orientation", "limited_tissue", "cautery_artifact",
    ],
    "Other differential": [],
}

PAIR_DIRECTIONS = {
    "HP vs SSL": ["supports_hp", "supports_ssl", "supports_both", "conflicts_final", "uncertain"],
    "SSL vs TSA": ["supports_ssl", "supports_tsa", "supports_both", "conflicts_final", "uncertain"],
    "TA vs TVA": ["supports_ta", "supports_tva", "supports_both", "conflicts_final", "uncertain"],
    "Adenoma vs inflammatory": [
        "supports_adenoma", "supports_inflammatory", "supports_both", "conflicts_final", "uncertain",
    ],
}
HGD_DIRECTIONS = ["definite_hgd", "suspicious_insufficient", "no_definite_hgd", "not_assessable"]
GENERAL_DIRECTIONS = [
    "supports_final", "supports_alternative", "non_discriminative", "conflicts_final", "uncertain",
]


def taxonomy_payload():
    directions = {challenge: PAIR_DIRECTIONS.get(challenge, GENERAL_DIRECTIONS) for challenge in PRIMARY_CHALLENGES}
    directions["Focal HGD"] = HGD_DIRECTIONS
    directions["Borderline HGD"] = HGD_DIRECTIONS
    return {
        "lesion_diagnoses": LESION_DIAGNOSES,
        "hgd_statuses": HGD_STATUSES,
        "label_actions": LABEL_ACTIONS,
        "challenge_dispositions": CHALLENGE_DISPOSITIONS,
        "primary_challenges": PRIMARY_CHALLENGES,
        "difficulty_modifiers": DIFFICULTY_MODIFIERS,
        "no_roi_reasons": NO_ROI_REASONS,
        "diagnostic_roles": DIAGNOSTIC_ROLES,
        "evidence_strengths": EVIDENCE_STRENGTHS,
        "general_evidence": GENERAL_EVIDENCE,
        "evidence_by_challenge": EVIDENCE_BY_CHALLENGE,
        "directions_by_challenge": directions,
    }
