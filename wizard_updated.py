# ui/wizard.py
# CICADAS-Compliant Dataset Description Wizard (TCIA)

from __future__ import annotations

import io
import shlex
import subprocess
import sqlite3
from datetime import datetime
from typing import Dict, List

import requests
import streamlit as st

OLLAMA_HOST = "http://127.0.0.1:11434"


# ----------------------------
# CICADAS section map (STRICT)
# ----------------------------
SECTIONS: List[dict] = [
    {
        "key": "title",
        "label": "Title",
        "guidance": "Full descriptive dataset title. Avoid abbreviations.",
        "required": True,
    },
    {
        "key": "abstract",
        "label": "Abstract",
        "guidance": "≤1000 characters. Include subject count, modalities, and intended use.",
        "required": True,
        "max_chars": 1000,
    },
    {
        "key": "introduction",
        "label": "Introduction",
        "guidance": "Scientific background, motivation, and uniqueness of the dataset.",
        "required": True,
    },
    {
        "key": "methods_subjects",
        "label": "Methods – Subject Selection",
        "guidance": "Inclusion/exclusion criteria, cohort definition, IRB status if applicable.",
        "required": True,
    },
    {
        "key": "methods_acquisition",
        "label": "Methods – Data Acquisition",
        "guidance": "Imaging modalities, scanners, protocols, resolution, acquisition parameters.",
        "required": True,
    },
    {
        "key": "methods_analysis",
        "label": "Methods – Data Analysis & Annotation",
        "guidance": "Preprocessing, labeling, annotation workflow, quality control.",
        "required": True,
    },
    {
        "key": "usage_notes",
        "label": "Usage Notes",
        "guidance": "Folder structure, caveats, known issues, recommended tools.",
        "required": True,
    },
    {
        "key": "external_resources",
        "label": "External Resources",
        "guidance": "Related publications, code repositories, external datasets (optional).",
        "required": False,
    },
]


# ----------------------------
# Session state
# ----------------------------
def _init_state():
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0

    if "wizard_answers" not in st.session_state:
        st.session_state.wizard_answers = {s["key"]: "" for s in SECTIONS}

    if "ai_rewrite" not in st.session_state:
        st.session_state.ai_rewrite = {s["key"]: "" for s in SECTIONS}


def _is_review(step: int) -> bool:
    return step >= len(SECTIONS)


# ----------------------------
# Ollama helper
# ----------------------------
def call_ollama(model: str, prompt: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
    except Exception:
        pass

    cmd = f"ollama run {shlex.quote(model)}"
    result = subprocess.run(
        cmd,
        input=prompt.encode(),
        capture_output=True,
        shell=True,
    )
    return result.stdout.decode(errors="ignore").strip()


def sanitize_ai_reply(text: str) -> str:
    """Strip common explanatory prefixes from the model's reply.

    Ollama models often prepend things like "Here is the rewritten section:" or
    "Below is the revised version:" before the actual rewrite.  This helper
    removes any such boilerplate so we store just the cleaned section.
    """
    if not text:
        return text
    import re
    # Look for prefixes like "Here is the rewritten section:" and strip them.
    # Use DOTALL so the pattern can match across lines if needed.
    patterns = [
        r"(?i).*?rewritten? section:\s*",
        r"(?i).*?revised version:\s*",
        r"(?i).*?revision:\s*",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            text = text[m.end():]
            break
    return text.strip()

# ----------------------------
# Original cicadas_rewrite_prompt (kept for reference):
# def cicadas_rewrite_prompt(section: dict, text: str) -> str:
#     return f"""..."""

# Fixed prompt generator that uses "Original section" and logs the prompt
# to help diagnose mysterious responses from Ollama.
def cicadas_rewrite_prompt_fixed(section: dict, text: str) -> str:
    prompt = f"""
You are preparing a dataset description for submission to The Cancer Imaging Archive (TCIA).

Rewrite ONLY the section below to strictly comply with CICADAS expectations.

Rules:
- Do NOT invent or assume information
- Use clear technical language
- Remove vague phrases
- Preserve factual content
- Use short paragraphs or bullet points where appropriate

Section: {section['label']}
Guidance: {section['guidance']}

Original section:
\"\"\"{text}\"\"\"

Rewritten CICADAS-compliant version:
""".strip()
    print("[DEBUG] cicadas rewrite prompt:\n", prompt)
    return prompt

# ----------------------------
# SQLite export
# ----------------------------
def build_sqlite_db_bytes(answers: Dict[str, str]) -> bytes:
    buf = io.BytesIO()
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    columns = ", ".join(f"{s['key']} TEXT" for s in SECTIONS)

    cur.execute(
        f"""
        CREATE TABLE cicadas_dataset_description (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            {columns}
        )
        """
    )

    cur.execute(
        f"""
        INSERT INTO cicadas_dataset_description (
            created_at,
            {", ".join(s["key"] for s in SECTIONS)}
        )
        VALUES (
            ?,
            {", ".join("?" for _ in SECTIONS)}
        )
        """,
        [datetime.now().isoformat()] + [answers[s["key"]] for s in SECTIONS],
    )

    conn.commit()

    for line in conn.iterdump():
        buf.write(f"{line}\n".encode())

    conn.close()
    buf.seek(0)
    return buf.getvalue()


# ----------------------------
# Main UI
# ----------------------------
def wizard_page():
    _init_state()
    step = st.session_state.wizard_step

    st.title("CICADAS Dataset Description Wizard")

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
        height=220,
    )

    if "max_chars" in section:
        st.caption(f"Characters: {len(text)} / {section['max_chars']}")

    st.session_state.wizard_answers[key] = text

    st.markdown("### AI rewrite (CICADAS-compliant)")
    model = st.text_input("Ollama model", value="llama3.2:3b")

    if st.button("Get AI feedback"):
        if text.strip():
            with st.spinner("Generating CICADAS rewrite..."):
                prompt = cicadas_rewrite_prompt_fixed(section, text)
                raw = call_ollama(model, prompt)
                st.session_state.ai_rewrite[key] = sanitize_ai_reply(raw)
        else:
            st.warning("Add text before requesting AI rewrite.")

    ai_text = st.session_state.ai_rewrite[key]

    if ai_text:
        st.text_area("AI revised version", ai_text, height=220)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Accept AI version"):
                st.session_state.wizard_answers[key] = ai_text
                st.session_state.ai_rewrite[key] = ""
                st.session_state.wizard_step += 1
                st.rerun()

        with col_b:
            if st.button("Reject AI version"):
                st.session_state.ai_rewrite[key] = ""
                st.session_state.wizard_step += 1
                st.rerun()

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

    for s in SECTIONS:
        st.markdown(f"## {s['label']}")
        st.write(st.session_state.wizard_answers[s["key"]] or "[Not provided]")

    st.download_button(
        "Download CICADAS responses (.db)",
        build_sqlite_db_bytes(st.session_state.wizard_answers),
        file_name="cicadas_dataset_description.db",
        mime="application/x-sqlite3",
    )


if __name__ == "__main__":
    wizard_page()
