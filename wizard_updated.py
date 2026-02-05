
# ui/wizard.py
# Dataset Description Wizard (CICADAS-style) for TCIA submissions
#
# Self-contained Streamlit wizard with:
# - step-by-step drafting
# - optional local AI feedback (Ollama)
# - session-state persistence
# - export to SQLite database (.db)
#
# NOTE: DOCX export has been intentionally removed.

from __future__ import annotations

import io
import json
import shlex
import subprocess
import sqlite3
from datetime import datetime
from typing import Dict, List

import requests
import streamlit as st

OLLAMA_HOST = "http://127.0.0.1:11434"


# ----------------------------
# Wizard section map
# ----------------------------
SECTIONS: List[dict] = [
    {
        "key": "title",
        "label": "Title",
        "guidance": "A clear, descriptive dataset name. Avoid vague titles.",
        "placeholder": "Example: Multi-institutional CT imaging dataset for lung nodule detection (2018–2023)",
        "required": True,
    },
    {
        "key": "abstract",
        "label": "Abstract",
        "guidance": "2–5 sentences summarizing the dataset and its purpose.",
        "placeholder": "High-level description of the dataset.",
        "required": True,
    },
    {
        "key": "background",
        "label": "Background / Rationale",
        "guidance": "Why the dataset exists and what gap it addresses.",
        "placeholder": "Scientific or clinical motivation.",
        "required": False,
    },
    {
        "key": "contents",
        "label": "Dataset Contents",
        "guidance": "What files and formats are included.",
        "placeholder": "DICOM, CSV, segmentations, reports, etc.",
        "required": True,
    },
    {
        "key": "cohort",
        "label": "Cohort / Subjects",
        "guidance": "Who or what is represented in the dataset.",
        "placeholder": "Inclusion/exclusion criteria at a high level.",
        "required": True,
    },
    {
        "key": "labels",
        "label": "Labels / Outcomes / Annotations",
        "guidance": "Describe labels and how they were generated.",
        "placeholder": "Annotation process and label definitions.",
        "required": False,
    },
]


# ----------------------------
# Session state initialization
# ----------------------------
def _init_state():
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0

    if "wizard_answers" not in st.session_state:
        st.session_state.wizard_answers = {s["key"]: "" for s in SECTIONS}

    if "wizard_feedback" not in st.session_state:
        st.session_state.wizard_feedback = {s["key"]: "" for s in SECTIONS}

    if "prefill" not in st.session_state:
        st.session_state.prefill = {
            "proposal_type": "new_collection",
            "email": "",
            "dataset_title": "",
            "dataset_description": "",
        }


def _total_steps() -> int:
    return len(SECTIONS) + 1


def _is_review(step: int) -> bool:
    return step >= len(SECTIONS)


# ----------------------------
# Ollama helper
# ----------------------------
def call_ollama(model: str, prompt: str, temperature: float = 0.2, num_predict: int = 450) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": num_predict},
                "stream": False,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            return (data.get("message", {}).get("content") or data.get("response") or "").strip()
    except Exception:
        pass

    try:
        cmd = f"ollama run {shlex.quote(model)}"
        result = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=180,
            shell=True,
        )
        return result.stdout.decode(errors="ignore").strip()
    except Exception as e:
        return f"[Ollama error] {e}"


def feedback_prompt(label: str, text: str) -> str:
    return f"""
You are an expert technical editor helping an investigator write a dataset description.

Section: {label}

Rules:
- Do not invent details.
- Ask for missing information explicitly.
- Be concise and professional.

Output format:
Missing:
Suggestions:
Rewrite:

Text:
\"\"\"{text}\"\"\"
""".strip()


# ----------------------------
# Compilation helpers
# ----------------------------
def compile_for_submit(answers: Dict[str, str]) -> str:
    blocks = []
    for s in SECTIONS:
        val = (answers.get(s["key"]) or "").strip()
        if val:
            blocks.append(f"{s['label']}:\n{val}\n")
    return "\n".join(blocks).strip()


# ----------------------------
# SQLite export
# ----------------------------
def build_sqlite_db_bytes(answers: Dict[str, str]) -> bytes:
    """
    Build an in-memory SQLite database containing the wizard responses
    and return it as raw bytes for download.
    """
    buf = io.BytesIO()
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    columns = ", ".join(f"{s['key']} TEXT" for s in SECTIONS)

    cur.execute(
        f"""
        CREATE TABLE dataset_description (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            {columns}
        )
        """
    )

    values = [answers.get(s["key"], "") for s in SECTIONS]

    cur.execute(
        f"""
        INSERT INTO dataset_description (
            created_at,
            {", ".join(s["key"] for s in SECTIONS)}
        )
        VALUES (
            ?,
            {", ".join("?" for _ in SECTIONS)}
        )
        """,
        [datetime.now().isoformat()] + values,
    )

    conn.commit()

    for line in conn.iterdump():
        buf.write(f"{line}\n".encode("utf-8"))

    conn.close()
    buf.seek(0)
    return buf.getvalue()


# ----------------------------
# Utilities
# ----------------------------
def _safe_filename(name: str) -> str:
    keep = [c for c in name if c.isalnum() or c in (" ", "_", "-")]
    return "".join(keep).strip().replace(" ", "_")[:80] or "dataset_description"


# ----------------------------
# Main UI
# ----------------------------
def wizard_page():
    _init_state()

    st.title("Dataset Description Wizard")
    st.caption("Step-by-step CICADAS-style drafting with export to SQLite.")

    step = st.session_state.wizard_step

    if _is_review(step):
        _render_review()
        return

    section = SECTIONS[step]
    key = section["key"]

    st.subheader(section["label"])
    st.info(section["guidance"])

    text = st.text_area(
        "Your draft",
        value=st.session_state.wizard_answers[key],
        placeholder=section["placeholder"],
        height=220,
    )

    st.session_state.wizard_answers[key] = text

    st.markdown("#### AI feedback (optional)")
    model = st.text_input("Local model", value="llama3.2:3b")

    if st.button("Get AI feedback"):
        if text.strip():
            prompt = feedback_prompt(section["label"], text)
            with st.spinner("Running AI feedback..."):
                st.session_state.wizard_feedback[key] = call_ollama(model, prompt)
        else:
            st.warning("Add text before requesting feedback.")

    if st.session_state.wizard_feedback[key]:
        st.text_area("Feedback", st.session_state.wizard_feedback[key], height=200)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back", disabled=step == 0):
            st.session_state.wizard_step -= 1
            st.rerun()

    with col2:
        disabled = section["required"] and not text.strip()
        if st.button("Next", disabled=disabled):
            st.session_state.wizard_step += 1
            st.rerun()


def _render_review():
    st.subheader("Review & Export")

    answers = st.session_state.wizard_answers

    for s in SECTIONS:
        st.markdown(f"### {s['label']}")
        st.write(answers.get(s["key"]) or "[TBD]")

    compiled = compile_for_submit(answers)
    st.markdown("### Combined text")
    st.text_area("Combined", compiled, height=200)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Back"):
            st.session_state.wizard_step = len(SECTIONS) - 1
            st.rerun()

    with col2:
        if st.button("Prefill Submit"):
            st.session_state.prefill["dataset_title"] = answers.get("title", "")
            st.session_state.prefill["dataset_description"] = compiled
            st.success("Submit page prefilled.")

    with col3:
        safe = _safe_filename(answers.get("title", "dataset_description"))

        st.download_button(
            "Download responses (.db)",
            build_sqlite_db_bytes(answers),
            file_name=f"{safe}.db",
            mime="application/x-sqlite3",
        )


if __name__ == "__main__":
    wizard_page()
#```
