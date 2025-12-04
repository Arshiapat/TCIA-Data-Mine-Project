# intake/form_utils.py
from __future__ import annotations
import io
import json

# ---- Controlled vocab ----
VOCAB_MODALITIES = ["CT", "MR", "PET", "US", "XR", "Pathology", "Genomics", "Clinical", "Other"]
VOCAB_PROPOSAL_TYPES = ["new_collection", "analysis_results"]

# ---- Defaults / Required ----
DEFAULTS = {
    "proposal_type": "new_collection",
    "title": "",
    "pi_name": "",
    "contact_email": "",
    "org_name": "",
    "data_modalities": [],
    "data_modalities_other": "",
    "short_abstract": "",
}

REQUIRED = [
    "proposal_type",
    "title",
    "pi_name",
    "contact_email",
    "org_name",
    "data_modalities",
    "short_abstract",
]

def _first_present(raw: dict, keys: list[str], default=None):
    for k in keys:
        if k in raw and raw[k] not in (None, ""):
            return raw[k]
    return default

def normalize_record(raw: dict) -> dict:
    """Map a user file (json/csv/yaml) into our canonical keys & formats."""
    rec = {**DEFAULTS}

    rec["proposal_type"] = _first_present(raw, ["proposal_type", "type", "proposalType"], DEFAULTS["proposal_type"])
    rec["title"] = _first_present(raw, ["title", "project_title", "name"], "")
    rec["pi_name"] = _first_present(raw, ["pi_name", "pi", "principal_investigator"], "")
    rec["contact_email"] = _first_present(raw, ["contact_email", "email"], "")
    rec["org_name"] = _first_present(raw, ["org_name", "org", "organization", "affiliation"], "")
    rec["data_modalities"] = _first_present(raw, ["data_modalities", "modalities", "modality"], [])
    rec["short_abstract"] = _first_present(raw, ["short_abstract", "abstract", "summary"], "")

    # proposal_type -> canonical
    if isinstance(rec["proposal_type"], str):
        v = rec["proposal_type"].strip().lower().replace(" ", "_")
        if v in {"new", "new_collection"}:
            rec["proposal_type"] = "new_collection"
        elif v in {"analysis", "analysis_results", "results"}:
            rec["proposal_type"] = "analysis_results"

    # modalities: accept str/list/tuple/set
    mods = rec.get("data_modalities", [])
    if isinstance(mods, str):
        parts = [p.strip() for p in mods.replace(";", ",").split(",") if p.strip()]
    elif isinstance(mods, (list, tuple, set)):
        parts = [str(p).strip() for p in mods if str(p).strip()]
    else:
        parts = []

    chosen, others = [], []
    vocab_lc = {m.lower(): m for m in VOCAB_MODALITIES}
    for p in parts:
        key = p.lower()
        if key in vocab_lc and key != "other":
            chosen.append(vocab_lc[key])
        elif key == "other":
            # literal "Other"
            pass
        else:
            others.append(p)

    rec["data_modalities"] = sorted(set(chosen + (["Other"] if others else [])))
    rec["data_modalities_other"] = ", ".join(sorted(set(others)))

    return rec

def validate(rec: dict) -> list[str]:
    errs: list[str] = []

    for k in REQUIRED:
        v = rec.get(k)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
            errs.append(f"Missing required field: {k}")

    if rec.get("proposal_type") not in VOCAB_PROPOSAL_TYPES:
        errs.append("proposal_type must be one of: new_collection, analysis_results")

    if "Other" in rec.get("data_modalities", []) and not rec.get("data_modalities_other", "").strip():
        errs.append('Please describe "Other" data_modalities.')

    em = rec.get("contact_email", "")
    if "@" not in em or "." not in em.split("@")[-1]:
        errs.append("contact_email looks invalid.")

    return errs

def parse_uploaded(file_like) -> dict:
    """
    Parse an uploaded file (Streamlit-style or BytesIO) into a raw dict.
    Supports .json, .csv, .yml/.yaml. Tries JSON as a fallback.
    """
    name = (getattr(file_like, "name", "") or "").lower()
    data = None

    # YAML optional
    try:
        import yaml  # type: ignore
        HAVE_YAML = True
    except Exception:
        HAVE_YAML = False

    if name.endswith(".json"):
        data = json.load(file_like)
    elif name.endswith((".yml", ".yaml")) and HAVE_YAML:
        data = yaml.safe_load(file_like.read())
    elif name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(file_like)
        if not df.empty:
            data = df.iloc[0].to_dict()
        else:
            data = {}
    else:
        # Try JSON fallback
        try:
            if isinstance(file_like, (io.BytesIO, io.StringIO)):
                # rewind to start if necessary
                try:
                    file_like.seek(0)
                except Exception:
                    pass
            data = json.load(file_like)
        except Exception:
            data = {}

    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    return data if isinstance(data, dict) else {}

# Optional CLI smoke test: python -m intake.form_utils sample_inputs/valid.json
if __name__ == "__main__":
    import sys
    from pathlib import Path
    p = Path(sys.argv[1])
    with p.open("rb") as f:
        class _F(io.BytesIO):
            def __init__(self, b, name): super().__init__(b); self.name = name
        buf = _F(f.read(), p.name)
    raw = parse_uploaded(buf)
    rec = normalize_record(raw)
    out = {"record": rec, "errors": validate(rec)}
    print(json.dumps(out, indent=2))
