# intake/intakeNew.py

from __future__ import annotations

import os, sys
# add the project root (one folder above "intake") to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, time
from datetime import datetime, timezone
import streamlit as st

from intake.form_utils import (
    parse_uploaded, normalize_record, validate,
    DEFAULTS, VOCAB_MODALITIES, VOCAB_PROPOSAL_TYPES
)
from intake.saver import save_record_to_parquet


APP_TITLE = "TCIA Unified Proposal – Submitter"
DATA_DIR = os.getenv("DATA_DIR", "./data/parquet")
SUBMIT_COOLDOWN_SEC = int(os.getenv("SUBMIT_COOLDOWN", "10"))

st.set_page_config(page_title=APP_TITLE, page_icon="📤", layout="centered")
os.makedirs(DATA_DIR, exist_ok=True)

if "form" not in st.session_state:
    st.session_state.form = {**DEFAULTS}
if "last_submit_ts" not in st.session_state:
    st.session_state.last_submit_ts = 0

st.title(APP_TITLE)
st.caption("Upload a file to auto-fill, review the fields, then submit. All submissions are append-only and partitioned by proposal type and date.")

# ---------- Upload → auto-fill ----------
st.subheader("1) Optional: Load from file")
uploaded = st.file_uploader("Upload JSON / CSV / YAML", type=["json", "csv", "yml", "yaml"])
if uploaded:
    raw = parse_uploaded(uploaded)
    norm = normalize_record(raw)
    st.session_state.form.update(norm)
    st.success("Loaded values from file. Review them below before submitting.")

st.markdown("---")

# ---------- Form ----------
st.subheader("2) Complete the form")
with st.form("submitter_form", clear_on_submit=False):
    # proposal type
    pt_default = st.session_state.form.get("proposal_type", DEFAULTS["proposal_type"])
    pt_index = VOCAB_PROPOSAL_TYPES.index(pt_default) if pt_default in VOCAB_PROPOSAL_TYPES else 0
    proposal_type = st.selectbox("Proposal Type *", VOCAB_PROPOSAL_TYPES, index=pt_index)

    title = st.text_input("Project Title *", value=st.session_state.form.get("title", ""))
    pi_name = st.text_input("Principal Investigator *", value=st.session_state.form.get("pi_name", ""))
    contact_email = st.text_input("Contact Email *", value=st.session_state.form.get("contact_email", ""))
    org_name = st.text_input("Organization *", value=st.session_state.form.get("org_name", ""))

    # controlled vocab + "Other"
    default_mods = [m for m in st.session_state.form.get("data_modalities", []) if m in VOCAB_MODALITIES]
    data_modalities = st.multiselect("Data Modalities *", options=VOCAB_MODALITIES, default=default_mods)
    data_modalities_other = st.text_input('If you chose "Other", describe it', value=st.session_state.form.get("data_modalities_other", ""))

    short_abstract = st.text_area("Short Abstract *", value=st.session_state.form.get("short_abstract", ""), height=160)

    submitted = st.form_submit_button("Submit Proposal", use_container_width=True)

if not submitted:
    st.stop()

# ---------- Cooldown ----------
now = time.time()
if now - st.session_state.last_submit_ts < SUBMIT_COOLDOWN_SEC:
    st.warning("Submitting too quickly — please wait a few seconds and try again.")
    st.stop()
st.session_state.last_submit_ts = now

# ---------- Validate ----------
record = {
    "proposal_type": proposal_type,
    "title": title,
    "pi_name": pi_name,
    "contact_email": contact_email,
    "org_name": org_name,
    "data_modalities": data_modalities,
    "data_modalities_other": data_modalities_other,
    "short_abstract": short_abstract,
}

errors = validate(record)
if errors:
    st.error("Please fix the issues below:\n\n" + "\n".join(f"• {e}" for e in errors))
    st.stop()

# ---------- Save (Parquet dataset) ----------
save_record_to_parquet(record, root=DATA_DIR)

# keep latest good state in session
st.session_state.form = {**record}

# ---------- Receipt ----------
receipt = {
    **record,
    "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
st.success("Submission saved successfully ✅")
st.download_button(
    "Download Receipt (JSON)",
    data=json.dumps(receipt, indent=2),
    file_name=f"tcia_submission_{receipt['saved_at_utc'].replace(':','-')}.json",
    use_container_width=True,
)

st.info(f"Data directory: `{DATA_DIR}` (partitioned by proposal_type and dt).")
