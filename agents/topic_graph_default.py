"""Autonomy topic tree (Parent/Children); shared by counselor navigation and client docs."""

DEFAULT_TOPIC_GRAPH = {
    # --- ROOT LEVEL ---
    "Autonomy": {
        "Parent": [],
        "Children": [
            "Sense of Choice",
            "Perceived Locus of Causality",
            "Control of Exercise Plan",
            "Consistent with Personal Identity",
            "Alignment with Personal Value",
            "No External Pressure",
            "Absence of Conditional Punishment",
            "Free of External Awards",
            "Driven by Health Outcomes",
        ],
    },
    # --- PRIMARY SUB-NODES ---
    "Sense of Choice": {"Parent": ["Autonomy"], "Children": []},
    "Perceived Locus of Causality": {"Parent": ["Autonomy"], "Children": []},
    "Control of Exercise Plan": {
        "Parent": ["Autonomy"],
        "Children": ["Time", "Place", "Type of Exercise"],
    },
    # --- LOGISTICAL NODES (under Control of Exercise Plan) ---
    "Time": {"Parent": ["Control of Exercise Plan"], "Children": []},
    "Place": {"Parent": ["Control of Exercise Plan"], "Children": []},
    "Type of Exercise": {"Parent": ["Control of Exercise Plan"], "Children": []},
    # --- PSYCHOLOGICAL & IDENTITY ALIGNMENT ---
    "Consistent with Personal Identity": {"Parent": ["Autonomy"], "Children": []},
    "Alignment with Personal Value": {"Parent": ["Autonomy"], "Children": []},
    # --- EXTERNAL PRESSURE & REGULATION ---
    "No External Pressure": {"Parent": ["Autonomy"], "Children": []},
    "Absence of Conditional Punishment": {"Parent": ["Autonomy"], "Children": []},
    "Free of External Awards": {"Parent": ["Autonomy"], "Children": []},
    "Driven by Health Outcomes": {"Parent": ["Autonomy"], "Children": []},
}

ROOT_TOPICS = ["Autonomy"]
