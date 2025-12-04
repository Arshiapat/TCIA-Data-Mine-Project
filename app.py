import os
import re
import io
import hmac
import time
import json
import uuid
import smtplib
import hashlib
import textwrap
import shlex
import subprocess
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import streamlit as st
import sqlite3

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from tcia_submission_analyzer import TCIADatasetAnalyzer

# --- local package import shim (MUST be before other imports) ---
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

# Helpers for auto-fill
from intake.form_utils import parse_uploaded, normalize_record, DEFAULTS

# ----------------------------
# Config & Utility
# ----------------------------
APP_TITLE = "TCIA Proposal Submissions"

# Parquet storage
DATA_DIR = os.getenv("DATA_DIR", "./data/parquet")

# SQLite storage (for admin viewer)
DB_PATH = os.path.join("data", "submissions.db")

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "change-me")
ADMIN_PIN = os.getenv("ADMIN_PIN", "123456")
ADMIN_IP_ALLOWLIST = [
    ip.strip()
    for ip in os.getenv("ADMIN_IP_ALLOWLIST", "").split(",")
    if ip.strip()
]

EMAIL_FROM_ADDR = os.getenv("EMAIL_FROM_ADDR", "johntuck4234@gmail.com")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "TCIA Submissions")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
BRAND_LOGO_URL = os.getenv("BRAND_LOGO_URL")

# Cooldowns / throttling (seconds)
SUBMIT_COOLDOWN = 20

st.set_page_config(page_title=APP_TITLE, page_icon="📤", layout="wide")

# Session state init
if "last_submit_ts" not in st.session_state:
    st.session_state.last_submit_ts = 0
if "admin_authed_until" not in st.session_state:
    st.session_state.admin_authed_until = 0
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None

# Auto-fill state (what the upload populates)
if "prefill" not in st.session_state:
    st.session_state.prefill = {
        "proposal_type": DEFAULTS.get("proposal_type", "new_collection"),
        "email": "",
        "dataset_title": "",
        "dataset_description": "",
    }

# Ensure data dir
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Controlled vocab
# ----------------------------
IMAGE_TYPES = [
    "MR",
    "CT",
    "PET",
    "PET-CT",
    "PET-MR",
    "Mammograms",
    "Ultrasound",
    "Xray",
    "Radiation Therapy (RTSTRUCT, RTDOSE, RTPLAN)",
    "Whole Slide Image",
    "CODEX",
    "Single-cell Image",
    "Photomicrograph",
    "Microarray",
    "Multiphoton",
    "Immunofluorescence",
]
SUPPORTING_DATA = [
    "Clinical (e.g. demographics, medical history/comorbidities, treatment details, outcomes)",
    "Image Analyses (e.g. segmentations, radiologist/pathologist reports, image features)",
    "Image Registrations",
    "Genomics (e.g. gene expression, copy number variation, methylation)",
    "Proteomics (e.g. mass spectrometry, protein arrays)",
    "Software / Source Code",
    "No additional data (only images)",
]
ANALYSIS_DERIVED_TYPES = [
    "Segmentation",
    "Classification",
    "Quantitative Feature",
    "Image (e.g. converted, processed or registered images)",
]
PUBLISH_REASONS = [
    "To meet a funding agency's data sharing requirements",
    "To meet a journal's data sharing requirements",
    "To facilitate a collaborative project with investigators outside my institution",
    "To facilitate a challenge competition",
]

# ----------------------------
# Validators
# ----------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[+\d][\d\s().-]{6,}$")
ORCID_RE = re.compile(r"^(\d{4}-){3}\d{3}[\dX]$")


def valid_email(x: str) -> bool:
    return bool(EMAIL_RE.match(x.strip())) if x else False


def valid_phone(x: str) -> bool:
    return bool(PHONE_RE.match(x.strip())) if x else False


def extract_orcids(text: str) -> list:
    return ORCID_RE.findall(text)


# ----------------------------
# Parquet engine helper
# ----------------------------
def parquet_engine() -> str:
    """Return pandas parquet engine ('pyarrow' or 'fastparquet')."""
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return "fastparquet"
        except Exception as e:
            raise RuntimeError(
                "Install either 'pyarrow' or 'fastparquet' to use Parquet."
            ) from e


# ----------------------------
# Storage: append-only parquet dataset
# ----------------------------
def write_record(proposal_type: str, record: dict):
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    part_dir = os.path.join(DATA_DIR, f"proposal_type={proposal_type}", f"dt={dt}")
    os.makedirs(part_dir, exist_ok=True)
    file_path = os.path.join(part_dir, "data.parquet")
    df_new = pd.DataFrame([record])
    if os.path.exists(file_path):
        df_existing = pd.read_parquet(file_path, engine=parquet_engine())
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.to_parquet(file_path, engine=parquet_engine(), index=False)
    else:
        df_new.to_parquet(file_path, engine=parquet_engine(), index=False)


def load_records(proposal_type: str, date: str | None):
    base = os.path.join(DATA_DIR, f"proposal_type={proposal_type}")
    if not os.path.isdir(base):
        return None

    paths = []
    if date:
        p = os.path.join(base, f"dt={date}", "data.parquet")
        if os.path.exists(p):
            paths.append(p)
    else:
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".parquet"):
                    paths.append(os.path.join(root, f))

    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p, engine=parquet_engine()))
        except Exception:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


# ----------------------------
# SQLite helper for admin
# ----------------------------
def query_submissions(proposal_type: str | None = None, date: str | None = None) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        q = "SELECT * FROM submissions"
        filters = []
        params = []
        if proposal_type:
            filters.append("proposal_type = ?")
            params.append(proposal_type)
        if date:
            filters.append("created_at LIKE ?")
            params.append(f"{date}%")
        if filters:
            q += " WHERE " + " AND ".join(filters)
        q += " ORDER BY created_at DESC"
        df = pd.read_sql_query(q, conn, params=params)
        return df
    finally:
        conn.close()


def run_dataset_analyzer_ui():
    st.subheader("TCIA Dataset Analyzer (local filesystem)")

    st.caption(
        "Point this at a local dataset folder on the server. "
        "This runs the enhanced analyzer and generates TCIA form helper answers."
    )

    dataset_path = st.text_input(
        "Dataset folder path on the server *",
        placeholder="/path/to/dataset/root",
        key="analyzer_dataset_path",
    )

    col1, col2 = st.columns(2)
    with col1:
        generate_html = st.checkbox(
            "Also prepare an HTML report for download", value=False
        )
    with col2:
        tcia_only = st.checkbox(
            "Only generate TCIA form answers (skip full console summary)",
            value=False,
        )

    if st.button("Run dataset analysis", use_container_width=True):
        path = dataset_path.strip()
        if not path:
            st.warning("Please enter a dataset path first.")
            return
        if not os.path.exists(path):
            st.error(f"Path does not exist on server: {path}")
            return
        if not os.path.isdir(path):
            st.error("The path must be a directory, not a file.")
            return

        with st.spinner("Analyzing dataset — this may take a while for large folders..."):
            analyzer = TCIADatasetAnalyzer(max_workers=4, verbose=False)
            results = analyzer.analyze_dataset(path)

        st.success("Analysis complete.")

        # Quick metrics
        overview = results.get("dataset_overview", {})
        dicom = results.get("dicom_catalog", {})
        tabular = results.get("tabular_catalog", {})

        st.markdown("### High-level metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total files", f"{overview.get('total_files', 0):,}")
        m2.metric("Size (GB)", f"{overview.get('total_size_gb', 0):.2f}")
        m3.metric(
            "DICOM patients",
            f"{dicom.get('hierarchy', {}).get('patient_count', 0)}",
        )
        m4.metric(
            "Tabular files",
            f"{tabular.get('file_count', 0)}",
        )

        # TCIA form auto-answers (text)
        tcia_text = analyzer._capture_print_output(analyzer.print_tcia_form_answers)
        st.markdown("### Auto-generated TCIA form answers")
        st.text_area(
            "These are the 6 high-confidence auto answers from the analyzer.",
            tcia_text,
            height=350,
        )

        # Optional full console-style summary (same as CLI output)
        if not tcia_only:
            summary_text = analyzer._capture_print_output(
                analyzer.print_comprehensive_summary
            )
        else:
            summary_text = ""
        combined_text = summary_text + ("\n\n" if summary_text else "") + tcia_text

        # Download buttons
        st.markdown("### Downloads")

        # JSON results (full structured output)
        json_bytes = json.dumps(results, indent=2, default=str).encode("utf-8")
        st.download_button(
            "Download JSON results",
            data=json_bytes,
            file_name="tcia_dataset_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

        # Text summary (what the CLI prints)
        st.download_button(
            "Download text summary",
            data=combined_text.encode("utf-8"),
            file_name="tcia_dataset_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Optional HTML report via the analyzer’s own HTML generator
        if generate_html:
            # Let the analyzer write an HTML file, then read it back for download
            html_path = analyzer.save_results(output_path=None, format="html")
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_bytes = f.read().encode("utf-8")
                st.download_button(
                    "Download HTML report",
                    data=html_bytes,
                    file_name="tcia_dataset_analysis.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Could not read HTML report: {e}")


# ----------------------------
# Alerts (optional)
# ----------------------------
def send_slack(message: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=6)
    except Exception:
        pass


def send_email(
    to_addrs: list[str],
    subject: str,
    html_body: str,
    attachments: list[tuple[str, bytes]] | None = None,
):
    if not SMTP_HOST or not EMAIL_FROM_ADDR:
        return
    msg = MIMEMultipart()
    msg["From"] = f"{EMAIL_FROM_NAME} <{EMAIL_FROM_ADDR}>"
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    for name, data in (attachments or []):
        part = MIMEApplication(data, Name=name)
        part["Content-Disposition"] = f'attachment; filename="{name}"'
        msg.attach(part)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        if SMTP_USER and SMTP_PASS:
            s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(EMAIL_FROM_ADDR, to_addrs, msg.as_string())


# ----------------------------
# PDF receipt (optional)
# ----------------------------
def render_pdf_receipt(record: dict) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        width, height = LETTER
        y = height - 1 * inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1 * inch, y, "TCIA Proposal Submission Receipt")
        y -= 0.3 * inch
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, y, f"Submission ID: {record['submission_id']}")
        y -= 0.2 * inch
        c.drawString(1 * inch, y, f"Type: {record['proposal_type']}")
        y -= 0.2 * inch
        c.drawString(1 * inch, y, f"Created At (UTC): {record['created_at']}")
        y -= 0.3 * inch
        wrap = textwrap.wrap(
            f"Title: {record.get('dataset_title', '')}", width=90
        )
        for line in wrap:
            c.drawString(1 * inch, y, line)
            y -= 0.18 * inch
        y -= 0.1 * inch
        wrap = textwrap.wrap(f"Email: {record.get('email', '')}", width=90)
        for line in wrap:
            c.drawString(1 * inch, y, line)
            y -= 0.18 * inch
        c.showPage()
        c.save()
        return buf.getvalue()
    except Exception:
        return None


# ----------------------------
# Security helpers (Admin)
# ----------------------------
def sign_token(data: str, ttl_seconds: int = 1800) -> str:
    exp = int(time.time()) + ttl_seconds
    payload = f"{data}.{exp}"
    sig = hmac.new(TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> bool:
    try:
        data, exp, sig = token.rsplit(".", 2)
        payload = f"{data}.{exp}"
        sig_expected = hmac.new(
            TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, sig_expected):
            return False
        return int(exp) >= int(time.time())
    except Exception:
        return False


def client_ip() -> str:
    return st.session_state.get("client_ip", "unknown")


# ----------------------------
# CICADAS helpers and Ollama
# ----------------------------
OLLAMA_HOST = "http://127.0.0.1:11434"


def manual_cicadas_check(text: str):
    checks = {
        "Title": ["title", "dataset name"],
        "Abstract": ["subject", "number", "modality", "application", "abstract"],
        "Introduction": ["purpose", "background", "benefit", "introduction"],
        "Methods: Inclusion/Exclusion": ["inclusion", "exclusion", "criteria"],
        "Methods: Data Acquisition": [
            "acquisition",
            "scanner",
            "kvp",
            "te",
            "tr",
            "slice",
            "contrast",
        ],
        "Methods: Data Analysis": [
            "preprocessing",
            "annotation",
            "segmentation",
            "qc",
            "quality",
            "software",
            "version",
        ],
        "Usage Notes": [
            "organization",
            "naming",
            "subset",
            "split",
            "software",
            "format",
        ],
        "External Resources": ["repository", "github", "tool", "dataset", "resource"],
        "Summary": ["summary", "conclusion", "value"],
    }
    low = text.lower()
    report = []
    all_ok = True
    for section, kws in checks.items():
        present = any(k in low for k in kws)
        if not present:
            all_ok = False
        report.append((section, present, kws))
    return all_ok, report


def cicadas_prompt(description_text: str, report):
    missing = [s for s, ok, _ in report if not ok]
    present = [s for s, ok, _ in report if ok]

    return f"""
You are a technical writer formatting a dataset description according to the CICADAS checklist.

CICADAS sections:
- Title
- Abstract
- Introduction
- Methods - Inclusion/Exclusion
- Methods - Data Acquisition
- Methods - Data Analysis
- Usage Notes
- External Resources
- Summary

Your task:
1. Read the author's text.
2. Use CICADAS as a guide to expand this into a fuller description.
3. For each section, write 1–3 concise sentences using clear, professional language.
4. Use reasonable domain knowledge for context (for example, why the dataset is useful, general goals), but do not invent specific numeric details, institution names, or scanner models that are not given.
5. If you truly have no basis for a section, write a short, helpful placeholder using "[TBD]" and explain what the author should add.
6. Do not include your internal reasoning, analysis, bullets, or JSON.
7. Do not use XML or HTML tags.
8. Do not use markdown headings or bullet lists.
9. Output only the final CICADAS-formatted description.

Formatting (use these labels exactly, one per line or small paragraph):

Title: ...
Abstract: ...
Introduction: ...
Methods - Inclusion/Exclusion: ...
Methods - Data Acquisition: ...
Methods - Data Analysis: ...
Usage Notes: ...
External Resources: ...
Summary: ...

Author text:
\"\"\"{description_text.strip()}\"\"\""""

def call_ollama(model: str, prompt: str, temperature: float = 0.2):
    """
    1) Try /api/chat
    2) Fallback to /api/generate with streaming
    3) Fallback to CLI
    """
    # 1) /api/chat
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": 400},
                "stream": False,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "response" in data:
                return data["response"].strip()
        elif resp.status_code != 404:
            resp.raise_for_status()
    except Exception:
        pass

    # 2) /api/generate streaming
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "options": {"temperature": temperature, "num_predict": 400},
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        if r.status_code == 200:
            chunks = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in obj and obj["error"]:
                    return f"[Ollama error] {obj['error']}"
                if "response" in obj:
                    chunks.append(obj["response"])
                elif "message" in obj and "content" in obj["message"]:
                    chunks.append(obj["message"]["content"])
            out = "".join(chunks).strip()
            return out or "[empty response]"
        elif r.status_code != 404:
            r.raise_for_status()
    except Exception:
        pass

    # 3) CLI fallback
    try:
        cmd = f"ollama run {shlex.quote(model)} --temperature {temperature}"
        result = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=180,
            shell=True,
        )
        if result.returncode != 0:
            return "[Ollama error] " + result.stderr.decode(errors="ignore").strip()
        return result.stdout.decode(errors="ignore").strip()
    except Exception as e:
        return f"[Ollama error] {e}"


# ----------------------------
# UI Components
# ----------------------------
def header():
    st.title(APP_TITLE)
    st.caption("Submit proposals for TCIA publication. Public form; admin view requires PIN.")


def sidebar_nav():
    return st.sidebar.radio(
        "Navigation",
        ["Submit", "Admin"],
        index=0,
    )


def basic_info_block():
    st.subheader("Basic Information")
    email = st.text_input(
        "Email *",
        value=st.session_state.prefill.get("email", ""),
        key="email",
    )
    sci_poc = st.text_area(
        "Provide a scientific point of contact *",
        placeholder="Name, email, phone",
        key="sci_poc",
    )
    tech_poc = st.text_area(
        "Provide a technical point of contact *",
        placeholder="Name, email, phone",
        key="tech_poc",
    )
    legal_admin = st.text_area(
        "Provide a legal/contracts administrator *",
        placeholder="Name, email",
        key="legal_admin",
    )
    time_constraints = st.text_area(
        "Are there any time constraints associated with sharing your data set? *",
        key="time_constraints",
    )
    return email, sci_poc, tech_poc, legal_admin, time_constraints


def dataset_pub_block(include_nickname_required: bool):
    st.subheader("Dataset Publication")

    title = st.text_input(
        "Suggest a descriptive title for your dataset *",
        value=st.session_state.prefill.get("dataset_title", ""),
        key="dataset_title",
    )

    nickname_label = "Suggest a shorter nickname for your dataset"
    if include_nickname_required:
        nickname_label += " *"

    nickname = st.text_input(
        nickname_label,
        help="< 30 chars; letters, numbers, dashes",
        key="dataset_nickname",
    )

    authors = st.text_area(
        "List the authors of this data set *",
        help="List as (FAMILY, GIVEN); include OrcIDs",
        key="authors",
    )

    desc_label = "Provide a Dataset Description *"
    description = st.text_area(
        desc_label,
        height=200,
        value=st.session_state.prefill.get("dataset_description", ""),
        key="dataset_description",
    )

    # CICADAS helper UI directly under the description
    with st.expander("CICADAS checklist and AI helper (optional)"):
        st.caption(
            "Run an explicit CICADAS check and optionally revise this description using a local model."
        )

        manual_clicked = st.form_submit_button(
            "Run CICADAS manual check", type="secondary"
        )
        revise_clicked = st.form_submit_button(
            "Revise description with AI", type="primary"
        )

        if manual_clicked:
            if not description.strip():
                st.warning(
                    "Please enter a dataset description above before running the CICADAS check."
                )
            else:
                all_ok, report = manual_cicadas_check(description)
                if all_ok:
                    st.success(
                        "CICADAS checklist looks satisfied based on a simple keyword scan."
                    )
                else:
                    st.error(
                        "Some CICADAS sections may be missing or incomplete."
                    )
                for section, present, kws in report:
                    if present:
                        st.success(f"{section}: detected")
                    else:
                        st.warning(f"{section}: missing or incomplete")
                        st.caption("Consider adding: " + ", ".join(kws))

    return title, nickname, authors, description, manual_clicked, revise_clicked


def data_collection_block():
    st.subheader("Data Collection Details")
    published_elsewhere = st.text_area(
        "Has this data ever been published elsewhere? *"
    )
    disease_site = st.text_input(
        "Primary disease site/location *",
        help="Use terms from provided document (Column A)",
    )
    histologic_dx = st.text_input(
        "Histologic diagnosis *",
        help="Use terms from provided document (Column B)",
    )

    image_types = st.multiselect(
        "Which image types are included in the data set? *", IMAGE_TYPES
    )
    image_types_other = st.text_input("Other image types (if any)")

    supporting_data = st.multiselect(
        "Which kinds of supporting data are included in the data set? *",
        SUPPORTING_DATA,
    )
    supporting_data_other = st.text_input("Other supporting data (if any)")

    file_formats = st.text_area(
        "Specify the file format utilized for each type of data *"
    )
    n_subjects = st.number_input(
        "How many subjects are in your data set? *", min_value=0, step=1
    )
    n_studies = st.text_input(
        "How many total radiology studies or pathology slides? *"
    )
    disk_space = st.text_input("Approximate disk space required *")
    preprocessing = st.text_area(
        "Steps taken to modify your data prior to submitting *"
    )

    faces = st.radio(
        "Does your data contain any images of patient faces? *", ["Yes", "No"]
    )

    usage_policy = st.radio(
        "Do you need to request any exceptions to TCIA's Open Access & Usage Policy? *",
        ["No exceptions requested", "Other"],
    )
    usage_policy_other = st.text_input(
        "If Other, please describe", disabled=(usage_policy == "No exceptions requested")
    )

    dataset_publications = st.text_area(
        "Publications specifically about dataset contents/how to use it (citations)"
    )
    additional_publications = st.text_area(
        "Additional publications where findings derived from these data were discussed"
    )
    acknowledgments = st.text_area(
        "Additional acknowledgments or funding statements *"
    )
    why_publish = st.text_area(
        "Why would you like to publish this dataset on TCIA? *"
    )

    return {
        "published_elsewhere": published_elsewhere,
        "disease_site": disease_site,
        "histologic_dx": histologic_dx,
        "image_types": image_types,
        "image_types_other": image_types_other,
        "supporting_data": supporting_data,
        "supporting_data_other": supporting_data_other,
        "file_formats": file_formats,
        "n_subjects": int(n_subjects),
        "n_studies": n_studies,
        "disk_space": disk_space,
        "preprocessing": preprocessing,
        "faces": faces,
        "usage_policy": usage_policy,
        "usage_policy_other": usage_policy_other,
        "dataset_publications": dataset_publications,
        "additional_publications": additional_publications,
        "acknowledgments": acknowledgments,
        "why_publish": why_publish,
    }


def analysis_only_block():
    st.subheader("Analysis-Specific Details")
    tcia_collections = st.text_input(
        "Which TCIA collection(s) did you analyze? *"
    )
    derived_types = st.multiselect(
        "What types of derived data are included? *", ANALYSIS_DERIVED_TYPES
    )
    derived_other = st.text_input("Other derived data type (if any)")
    n_patients = st.text_input(
        "How many patients? Include DICOM series or slide totals if known. *"
    )
    disk_space = st.text_input("Approximate disk space required *")
    have_records = st.radio(
        "Do you know exactly which TCIA images were analyzed? *",
        ["Yes", "No"],
    )
    file_formats = st.text_area(
        "Specify file format utilized for each type of data *"
    )
    primary_citation = st.text_area(
        "If there is a publication to cite when utilizing this data please provide the citation"
    )
    extra_pubs = st.text_area(
        "If there are any additional publications about this data set please list them"
    )
    acknowledgments = st.text_area(
        "Additional acknowledgements or funding statements *"
    )

    reasons = st.multiselect(
        "Why would you like to publish this dataset on TCIA? *",
        PUBLISH_REASONS,
    )
    reasons_other = st.text_input("Other reason (if any)")

    return {
        "tcia_collections": tcia_collections,
        "derived_types": derived_types,
        "derived_other": derived_other,
        "n_patients": n_patients,
        "disk_space": disk_space,
        "have_records": have_records,
        "file_formats": file_formats,
        "primary_citation": primary_citation,
        "extra_pubs": extra_pubs,
        "acknowledgments": acknowledgments,
        "reasons": reasons,
        "reasons_other": reasons_other,
    }


# ----------------------------
# Submit Page
# ----------------------------
def submit_page():
    header()

    # Upload to auto-fill
    st.subheader("Load from file (optional)")
    up = st.file_uploader(
        "Upload JSON / CSV / YAML to auto-fill core fields",
        type=["json", "csv", "yml", "yaml"],
    )
    if up:
        raw = parse_uploaded(up)
        norm = normalize_record(raw)
        st.session_state.prefill["proposal_type"] = norm.get(
            "proposal_type", st.session_state.prefill["proposal_type"]
        )
        st.session_state.prefill["email"] = norm.get(
            "contact_email", st.session_state.prefill["email"]
        )
        st.session_state.prefill["dataset_title"] = norm.get(
            "title", st.session_state.prefill["dataset_title"]
        )
        st.session_state.prefill["dataset_description"] = norm.get(
            "short_abstract", st.session_state.prefill["dataset_description"]
        )
        st.success("Loaded values from file. Review/edit below before submitting.")
    st.markdown("---")

    # Type selector
    default_pt = st.session_state.prefill.get("proposal_type", "new_collection")
    default_index = 0 if default_pt == "new_collection" else 1
    proposal_type = st.selectbox(
        "Proposal Type", ["New Collection", "Analysis Results"], index=default_index
    )
    st.info(f"You are filling out the {proposal_type} proposal.")
    st.markdown("---")

    # Honeypot
    st.markdown(
        "<div style='position:absolute;left:-10000px;' aria-hidden='true'>"
        "<input name='website' value='' />"
        "</div>",
        unsafe_allow_html=True,
    )
    honeypot = (
        st.query_params.get("website", "")
        if hasattr(st, "query_params")
        else ""
    )

    # Form
    with st.form("proposal_form", clear_on_submit=False):
        email, sci_poc, tech_poc, legal_admin, time_constraints = basic_info_block()
        (
            title,
            nickname,
            authors,
            description,
            manual_clicked,
            revise_clicked,
        ) = dataset_pub_block(
            include_nickname_required=(proposal_type == "New Collection")
        )

        if proposal_type == "New Collection":
            details = data_collection_block()
        else:
            details = analysis_only_block()

        st.markdown("---")
        st.subheader("Notification Options (optional)")
        alert_recipients = st.text_input(
            "Recipient email(s) for alerts (comma-separated)"
        )
        send_submitter_receipt = st.checkbox(
            "Email a receipt to the submitter", value=True
        )

        submitted = st.form_submit_button(
            "Submit Proposal", use_container_width=True
        )

    # Handle AI revision click without submit
    if revise_clicked and not submitted:
        if not description.strip():
            st.warning(
                "Please enter a dataset description before asking AI to revise it."
            )
            return

        all_ok, report = manual_cicadas_check(description)
        prompt = cicadas_prompt(description, report)

        with st.spinner(
            "Revising dataset description with AI (llama3.2:3b)..."
        ):
            ai_text = call_ollama("llama3.2:3b", prompt)

        st.session_state.prefill["dataset_description"] = ai_text
        st.success(
            "Description updated by AI. Review the new text above, then submit when ready."
        )
        st.experimental_rerun()

    if not submitted:
        return

    # Cooldown
    now = time.time()
    if now - st.session_state.get("last_submit_ts", 0) < SUBMIT_COOLDOWN:
        st.warning("You are submitting too fast. Please wait a few seconds.")
        return
    st.session_state["last_submit_ts"] = now

    # Validations (shared)
    errors = []
    if honeypot:
        errors.append("Spam detected.")
    if not valid_email(email):
        errors.append("Valid Email is required.")
    if not sci_poc.strip():
        errors.append("Scientific point of contact is required.")
    if not tech_poc.strip():
        errors.append("Technical point of contact is required.")
    if not legal_admin.strip():
        errors.append("Legal/contracts administrator is required.")
    if proposal_type == "New Collection" and not time_constraints.strip():
        errors.append(
            "Time constraints field is required for New Collection."
        )
    if not title.strip():
        errors.append("Dataset title is required.")
    if proposal_type == "New Collection":
        if not nickname.strip():
            errors.append("Dataset nickname is required for New Collection.")
        elif not re.fullmatch(r"[A-Za-z0-9\-]{1,30}", nickname or ""):
            errors.append(
                "Nickname must be <30 chars and only letters, numbers, or dashes."
            )
    if not authors.strip():
        errors.append("Authors are required.")
    if not description.strip():
        errors.append("Dataset description is required.")

    # Per-type validations
    if proposal_type == "New Collection":
        must = [
            (details["published_elsewhere"], "Published elsewhere is required."),
            (details["disease_site"], "Primary disease site is required."),
            (details["histologic_dx"], "Histologic diagnosis is required."),
            (details["image_types"], "At least one image type is required."),
            (
                details["supporting_data"],
                "At least one supporting data type is required.",
            ),
            (details["file_formats"], "File formats are required."),
            (details["n_studies"], "Total studies/slides is required."),
            (details["disk_space"], "Approximate disk space is required."),
            (details["preprocessing"], "Preprocessing steps are required."),
            (details["faces"], "Faces selection is required."),
            (details["usage_policy"], "Usage policy selection is required."),
            (details["acknowledgments"], "Acknowledgments are required."),
            (details["why_publish"], "Reason to publish is required."),
        ]
    else:
        must = [
            (
                details["tcia_collections"],
                "TCIA collections analyzed are required.",
            ),
            (
                details["derived_types"],
                "At least one derived data type is required.",
            ),
            (details["n_patients"], "Patient/series totals are required."),
            (details["disk_space"], "Approximate disk space is required."),
            (
                details["have_records"],
                "Record knowledge selection is required.",
            ),
            (details["file_formats"], "File formats are required."),
            (details["acknowledgments"], "Acknowledgments are required."),
            (details["reasons"], "At least one reason is required."),
        ]

    for val, msg in must:
        if (isinstance(val, list) and len(val) == 0) or (
            isinstance(val, str) and not val.strip()
        ):
            errors.append(msg)

    if errors:
        st.error("\n".join([f"• {e}" for e in errors]))
        return

    # Build record & save
    submission_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    record = {
        "submission_id": submission_id,
        "created_at": created_at,
        "proposal_type": proposal_type.replace(" ", "_").lower(),
        "email": email.strip(),
        "scientific_poc": sci_poc.strip(),
        "technical_poc": tech_poc.strip(),
        "legal_admin": legal_admin.strip(),
        "time_constraints": time_constraints.strip(),
        "dataset_title": title.strip(),
        "dataset_nickname": nickname.strip(),
        "authors": authors.strip(),
        "dataset_description": description.strip(),
        "client_ip": client_ip(),
        "user_agent": st.session_state.get("user_agent", "unknown"),
    }
    for k, v in details.items():
        record[k] = json.dumps(v) if isinstance(v, list) else v

    write_record(record["proposal_type"], record)

    pdf_bytes = render_pdf_receipt(record) or b""

    recipients = [
        x.strip()
        for x in (alert_recipients or "").split(",")
        if x.strip() and valid_email(x.strip())
    ]
    if recipients:
        send_email(
            recipients,
            subject=f"[TCIA] New {proposal_type} submission",
            html_body="<p>New submission received.</p><pre>"
            + json.dumps(record, indent=2)
            + "</pre>",
            attachments=[("receipt.pdf", pdf_bytes)] if pdf_bytes else None,
        )
    send_slack(f"New {proposal_type} submission: {submission_id}")

    if send_submitter_receipt and valid_email(record["email"]):
        send_email(
            [record["email"]],
            subject=f"TCIA Submission Receipt – {submission_id}",
            html_body="<p>Thank you for your submission.</p><pre>"
            + json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ["client_ip", "user_agent"]
                },
                indent=2,
            )
            + "</pre>",
            attachments=[("receipt.pdf", pdf_bytes)] if pdf_bytes else None,
        )

    st.success("Submission received! Your Submission ID is: " + submission_id)
    st.download_button(
        "Download Receipt (JSON)",
        json.dumps(record, indent=2),
        file_name=f"tcia_receipt_{submission_id}.json",
    )


# ----------------------------
# Admin Page - Parquet + SQLite
# ----------------------------
def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    # Optional IP allowlist
    if ADMIN_IP_ALLOWLIST:
        ip = client_ip()
        if ip not in ADMIN_IP_ALLOWLIST:
            st.error("Your IP is not allowlisted.")
            return

    col1, col2 = st.columns(2)
    with col1:
        pin = st.text_input("Enter Admin PIN", type="password")
    with col2:
        if st.button("Get Token", use_container_width=True):
            if pin == ADMIN_PIN:
                tok = sign_token("admin", ttl_seconds=1800)
                st.session_state.admin_token = tok
                st.session_state.admin_authed_until = time.time() + 1800
                st.success("Token issued. You have 30 minutes.")
            else:
                st.error("Invalid PIN")

    token = st.text_input(
        "Or paste an existing token",
        value=st.session_state.admin_token or "",
    )

    if not (token and verify_token(token)):
        st.info("Enter a valid PIN or token to proceed.")
        return

    st.markdown("---")
    admin_tool = st.radio(
        "Admin tool",
        ["Submissions viewer", "Dataset analyzer"],
        index=0,
    )

    if admin_tool == "Submissions viewer":
        backend = st.radio(
            "Storage backend",
            ["Parquet partitions", "SQLite database"],
            index=0,
        )

        if backend == "Parquet partitions":
            ptype = st.selectbox(
                "Proposal Type",
                ["new_collection", "analysis_results"],
                index=0,
            )
            date = st.text_input(
                "Date partition (YYYY-MM-DD) or leave blank for all", value=""
            )

            if st.button("Load Submissions", use_container_width=True):
                records_df = load_records(ptype, date)
                if records_df is None or records_df.empty:
                    st.warning("No submissions found for the selected filters.")
                else:
                    st.dataframe(records_df, use_container_width=True)
                    csv = records_df.to_csv(index=False).encode()
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name=f"tcia_{ptype}_{date or 'all'}.csv",
                    )
                    try:
                        buf = io.BytesIO()
                        records_df.to_parquet(
                            buf, engine=parquet_engine(), index=False
                        )
                        st.download_button(
                            "Download Parquet",
                            buf.getvalue(),
                            file_name=f"tcia_{ptype}_{date or 'all'}.parquet",
                        )
                    except Exception as e:
                        st.info("Parquet download not available: " + str(e))
        else:
            # SQLite admin view
            ptype = st.selectbox(
                "Proposal Type",
                ["", "new_collection", "analysis_results"],
                index=0,
            )
            date = st.text_input(
                "Filter by date (YYYY-MM-DD) or leave blank for all",
                value="",
            )

            if st.button(
                "Load Submissions (SQLite)", use_container_width=True
            ):
                df = query_submissions(
                    ptype if ptype else None, date if date else None
                )
                if df is None or df.empty:
                    st.warning("No submissions found for the selected filters.")
                    return
                st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False).encode()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"tcia_submissions_{ptype or 'all'}_{date or 'all'}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with col2:
                    if os.path.exists(DB_PATH):
                        with open(DB_PATH, "rb") as f:
                            db_bytes = f.read()
                        st.download_button(
                            label="Download Full Database (.db)",
                            data=db_bytes,
                            file_name="submissions.db",
                            mime="application/octet-stream",
                            use_container_width=True,
                        )
                    else:
                        st.info("SQLite database file not found at " + DB_PATH)
    else:
        # New dataset analyzer tool
        run_dataset_analyzer_ui()



# ----------------------------
# Main
# ----------------------------
nav = sidebar_nav()
if nav == "Submit":
    submit_page()
else:
    admin_page()
