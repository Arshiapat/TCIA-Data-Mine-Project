# tests/test_form_utils.py
import io
import json
import pytest

from intake.form_utils import (
    parse_uploaded, normalize_record, validate,
    DEFAULTS, VOCAB_MODALITIES, VOCAB_PROPOSAL_TYPES
)

class FakeUpload(io.BytesIO):
    """Mimic Streamlit's uploaded file: has .name and is a file-like object."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name

def test_parse_uploaded_json():
    payload = {
        "proposalType": "Analysis Results",
        "project_title": "Demo",
        "pi": "PI",
        "email": "pi@x.com"
    }
    f = FakeUpload(json.dumps(payload).encode(), "file.json")
    out = parse_uploaded(f)
    assert out["proposalType"] == "Analysis Results"
    assert out["project_title"] == "Demo"

def test_parse_uploaded_csv():
    csv_bytes = (
        b"type,project_title,pi,contact_email,org,modalities,abstract\n"
        b"new_collection,Proj,PI,pi@x.com,Org,\"CT, XR\",Abs\n"
    )
    f = FakeUpload(csv_bytes, "file.csv")
    out = parse_uploaded(f)
    assert out["project_title"] == "Proj"
    assert "modalities" in out

def test_parse_uploaded_yaml_if_available():
    try:
        import yaml  # noqa: F401
    except Exception:
        pytest.skip("pyyaml not installed")
    yml = b"proposal_type: new_collection\ntitle: Title\n"
    f = FakeUpload(yml, "file.yaml")
    out = parse_uploaded(f)
    assert out["proposal_type"] == "new_collection"

def test_normalize_record_maps_aliases_and_other_bucket():
    raw = {
        "proposalType": "Analysis Results",
        "project_title": "Liver Tumor",
        "pi": "Dr. Gray",
        "email": "g@u.edu",
        "organization": "Metro",
        "modalities": ["CT", "histology", "MR"],
        "summary": "..."
    }
    rec = normalize_record(raw)
    assert rec["proposal_type"] in {"analysis_results", "new_collection"}
    assert rec["title"] == "Liver Tumor"
    assert rec["pi_name"] == "Dr. Gray"
    assert rec["contact_email"] == "g@u.edu"
    assert set(rec["data_modalities"]).issuperset({"CT", "MR"})
    assert "Other" in rec["data_modalities"]
    assert "histology" in rec["data_modalities_other"]

def test_validate_rules():
    good = {
        "proposal_type": "new_collection",
        "title": "T",
        "pi_name": "P",
        "contact_email": "p@x.com",
        "org_name": "O",
        "data_modalities": ["CT"],
        "short_abstract": "A",
        "data_modalities_other": ""
    }
    assert validate(good) == []

    bad_email = {**good, "contact_email": "not-an-email"}
    errs = validate(bad_email)
    assert any("looks invalid" in e for e in errs)

    need_other = {**good, "data_modalities": ["Other"], "data_modalities_other": ""}
    errs = validate(need_other)
    assert any('describe "Other"' in e for e in errs)
