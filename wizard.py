# ui/wizard.py
# Dataset Description Wizard (CICADAS-style) for TCIA submissions
#
# Drop this file at: TCIA-Data-Mine-Project/ui/wizard.py
# Then in app.py:
#   from ui.wizard import wizard_page
#   nav = st.sidebar.radio("Navigation", ["Submit", "Dataset Wizard", "Admin"], index=0)
#   if nav == "Dataset Wizard": wizard_page()
#
# Notes:
# - AI output is treated as the rewrite directly (no "Rewrite:" parsing).
# - Accept replaces the current draft field immediately.
# - Dialog state is closed on Accept/Decline and on Back/Next to prevent bleed across sections.
# - Widget keys are the source of truth (no value= + session_state conflict).
# - DOCX export removed (prefill Submit page is the target flow).

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List

import requests
import streamlit as st

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# ----------------------------
# Wizard section map
# ----------------------------
SECTIONS: List[dict] = [
    {
        "key": "title",
        "label": "Title",
        "guidance": "A clear, descriptive dataset name. Avoid vague titles like 'Cancer Imaging Dataset'.",
        "placeholder": "Example: Multi-institutional CT imaging dataset for lung nodule detection (2018–2023)",
        "required": True,
    },
    {
        "key": "abstract",
        "label": "Abstract",
        "guidance": "2–5 sentences: what the dataset is, who/what it includes, modalities, and the intended purpose.",
        "placeholder": "Describe the dataset at a high level without deep method details.",
        "required": True,
    },
    {
        "key": "background",
        "label": "Background / Rationale",
        "guidance": "Why this dataset exists, the clinical/scientific motivation, and what gap it addresses.",
        "placeholder": "Explain the problem space and why sharing this dataset is useful.",
        "required": False,
    },
    {
        "key": "contents",
        "label": "Dataset Contents",
        "guidance": "What is included: DICOM, segmentations, labels, clinical tables, reports, code, etc.",
        "placeholder": "List the components and formats (DICOM, CSV, NIfTI, JSON, etc.).",
        "required": True,
    },
    {
        "key": "cohort",
        "label": "Cohort / Subjects",
        "guidance": "Who/what is represented, inclusion criteria, time range, and any key stratifications.",
        "placeholder": "Include inclusion/exclusion at a high level and what each subject represents.",
        "required": True,
    },
    {
        "key": "labels",
        "label": "Labels / Outcomes / Annotations",
        "guidance": "Describe ground truth: label definitions, who annotated, and what artifacts exist (segmentations, classes, etc.).",
        "placeholder": "Explain label meanings and how they were produced, without inventing details you do not have.",
        "required": False,
    },
]

# ----------------------------
# Ollama caller
# ----------------------------
def call_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.2,
    num_predict: int = 220,
    timeout_s: int = 120,
) -> str:
    """
    1) Try /api/chat (non-stream)
    2) Fallback to /api/generate (stream)
    3) Fallback to CLI (ollama run <model>)
    Returns a readable error string instead of failing silently.
    """

    model = (model or "").strip()
    if not model:
        return "[Ollama error] Model name is empty."

    options = {"temperature": temperature, "num_predict": num_predict, "num_ctx": 2048}

    # 1) /api/chat
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": options,
                "stream": False,
            },
            timeout=timeout_s,
        )

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                msg = data.get("message") or {}
                if isinstance(msg, dict) and "content" in msg:
                    return (msg.get("content") or "").strip()
                if "response" in data:
                    return (data.get("response") or "").strip()
            return "[Ollama error] Unexpected /api/chat response format."

        if resp.status_code != 404:
            return f"[Ollama error] /api/chat HTTP {resp.status_code}: {resp.text[:300]}"

    except Exception as e:
        chat_err = str(e)
    else:
        chat_err = None

    # 2) /api/generate (stream)
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "options": options,
                "stream": True,
            },
            stream=True,
            timeout=timeout_s,
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

                if obj.get("error"):
                    return f"[Ollama error] {obj['error']}"

                if "response" in obj and obj["response"] is not None:
                    chunks.append(obj["response"])

                if obj.get("done") is True:
                    break

            out = "".join(chunks).strip()
            return out or "[empty response]"

        if r.status_code != 404:
            return f"[Ollama error] /api/generate HTTP {r.status_code}: {r.text[:300]}"

    except Exception as e:
        gen_err = str(e)
    else:
        gen_err = None

    # 3) CLI fallback
    try:
        cmd = ["ollama", "run", model]
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            return "[Ollama error] " + (err or "CLI call failed.")
        return (result.stdout or "").strip()

    except Exception as e:
        cli_err = str(e)

    bits = []
    if chat_err:
        bits.append(f"chat: {chat_err}")
    if gen_err:
        bits.append(f"generate: {gen_err}")
    bits.append(f"host: {OLLAMA_HOST}")
    bits.append(f"cli: {cli_err}")
    return "[Ollama error] All methods failed. " + " | ".join(bits)


# ----------------------------
# Prompting
# ----------------------------
def _section_hard_constraints(section_key: str) -> str:
    if section_key == "title":
        return (
            "Hard constraints for Title:\n"
            "- Output a single line only.\n"
            "- Do NOT include labels like 'Dataset Name' or 'CICADAS Dataset Name'.\n"
            "- Do NOT use colons.\n"
            "- No bullets, no headings, no code blocks.\n"
        )
    if section_key == "abstract":
        return (
            "Hard constraints for Abstract:\n"
            "- Output 2 to 5 sentences as one paragraph.\n"
            "- No bullets.\n"
        )
    return ""


def feedback_prompt(section_key: str, section_label: str, section_text: str, section_guidance: str = "") -> str:
    section_text = (section_text or "").strip()
    section_guidance = (section_guidance or "").strip()
    hard = _section_hard_constraints(section_key)

    return f"""
You are an expert technical editor for The Cancer Imaging Archive (TCIA).
You are rewriting the user's text to be clearer, more specific, and more informative, using only the information they provided.

CICADAS is the standardized dataset description format used by TCIA to ensure clarity, reproducibility, and reuse of shared imaging datasets.

Section to improve: {section_label}

Section requirements:
{section_guidance if section_guidance else "Follow standard CICADAS-style expectations for this section."}

{hard}

STRICT RULES:
- Output ONLY the rewritten text for this section.
- Do NOT add explanations, commentary, or multiple options, or annotations.
- Do NOT add labels, headings, or formatting.
- Do NOT invent any specifics that are not in the original text. If details are missing, improve clarity using general language without adding new facts.
- Do NOT use Markdown or code blocks.
- Do NOT invent or assume specifics (counts, dates, scanner models, institutions, outcomes).
- If something is vague, improve clarity using general language, without adding new facts.
- The rewrite must not be less informative than the input.
- Keep content limited to this section only.
- Do NOT reference other sections or add content from other sections.
- Do NOT use colons in the Title section.
- Always keep the Title to a single line.
- For the Abstract, keep it to 2-5 sentences in a single paragraph, without bullets.
- Always follow the specific guidance and hard constraints for each section.
- If the input is very brief, the rewrite should expand it into a more complete description while adhering to the constraints above.
- Always ensure the output is in a polished, professional tone suitable for a scientific dataset description.
- Do NOT include any text that is not a direct rewrite of the user's input for this section.
- Do NOT copy and paste the user's input; the output should be a rewritten version that improves clarity and informativeness while adhering to the constraints above.
- Do NOT include any prefatory or concluding remarks, explanations, or commentary in the output. Only provide the rewritten section text.
- Do NOT include any labels, headings, or formatting in the output. Only provide the rewritten section text.
- Do NOT add any content that is not in the original text. If the original text is missing details, do not invent them; simply rewrite what is there to be clearer and more informative without adding new facts.
- Do NOT use Markdown formatting, bullets, or code blocks in the output. Only provide plain text.
- Always ensure the output is in a polished, professional tone suitable for a scientific dataset description.
- The output should be a rewritten version of the user's input for this section that improves clarity and informativeness while adhering to all the constraints above.


Text to rewrite:
\"\"\"{section_text}\"\"\"
""".strip()


def _clean_model_output(section_key: str, text: str) -> str:
    """Final safety net cleanup, especially for Title."""
    t = (text or "").strip()
    if not t:
        return ""

    # Remove code fences if the model ignores instructions
    t = t.replace("```", "").strip()

    # Remove common label prefixes
    prefixes = [
        "CICADAS Dataset Name:",
        "Dataset Name:",
        "CICADAS Title:",
        "Title:",
        "CICADAS Dataset Name -",
        "Dataset Name -",
        "CICADAS Title -",
        "Title -",
    ]
    for p in prefixes:
        if t.startswith(p):
            t = t[len(p):].strip()

    if section_key == "title":
        # Keep only first line
        t = t.splitlines()[0].strip()
        # If it still contains a colon label pattern, take the part after the last colon
        if ":" in t:
            t = t.split(":")[-1].strip()

    return t.strip()


def compile_for_submit(answers: Dict[str, str]) -> str:
    """Compile a single text blob suitable for pre-filling the Submit form's dataset_description."""
    blocks = []
    for s in SECTIONS:
        k = s["key"]
        val = (answers.get(k) or "").strip()
        if not val:
            continue
        blocks.append(f"{s['label']}:\n{val}\n")
    return "\n".join(blocks).strip()


# ----------------------------
# Session state helpers
# ----------------------------
def _close_feedback_dialog_state():
    st.session_state.wizard_show_feedback_dialog = False
    st.session_state.wizard_feedback_section_key = ""


def _init_state():
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0

    if "wizard_answers" not in st.session_state:
        st.session_state.wizard_answers = {s["key"]: "" for s in SECTIONS}

    if "wizard_feedback" not in st.session_state:
        st.session_state.wizard_feedback = {s["key"]: "" for s in SECTIONS}

    if "wizard_show_feedback_dialog" not in st.session_state:
        st.session_state.wizard_show_feedback_dialog = False

    if "wizard_feedback_section_key" not in st.session_state:
        st.session_state.wizard_feedback_section_key = ""

    if "prefill" not in st.session_state:
        st.session_state.prefill = {
            "proposal_type": "new_collection",
            "email": "",
            "dataset_title": "",
            "dataset_description": "",
        }

    # Widget keys are the source of truth
    for s in SECTIONS:
        k = s["key"]
        widget_key = f"wizard_input_{k}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state.wizard_answers.get(k, "")


def _current_total_steps() -> int:
    return len(SECTIONS) + 1


def _is_review_step(step_idx: int) -> bool:
    return step_idx >= len(SECTIONS)


# ----------------------------
# UI
# ----------------------------
def wizard_page():
    _init_state()

    if st.session_state.wizard_show_feedback_dialog and st.session_state.wizard_feedback_section_key:
        _feedback_dialog()

    st.title("Dataset Description Wizard")
    st.caption(
        "Step-by-step CICADAS-style drafting with local AI rewrite. "
        "At the end you can prefill the main Submit form."
    )

    # Sidebar helper panel
    with st.sidebar:
        st.markdown("### Wizard Progress")
        total = _current_total_steps()
        step = int(st.session_state.wizard_step)
        st.write(f"Step {min(step + 1, total)} of {total}")

        labels = [s["label"] for s in SECTIONS] + ["Review & Prefill"]
        for i, lab in enumerate(labels):
            marker = "➡️ " if i == step else ""
            st.write(f"{marker}{lab}")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reset wizard", use_container_width=True):
                _close_feedback_dialog_state()
                st.session_state.wizard_step = 0
                st.session_state.wizard_answers = {s["key"]: "" for s in SECTIONS}
                st.session_state.wizard_feedback = {s["key"]: "" for s in SECTIONS}
                for s in SECTIONS:
                    st.session_state[f"wizard_input_{s['key']}"] = ""
                st.rerun()

        with col_b:
            if st.button("Load example", use_container_width=True):
                _close_feedback_dialog_state()
                st.session_state.wizard_answers.update(
                    {
                        "title": "Example: Multi-site imaging dataset for [TBD] application",
                        "abstract": "This dataset contains [TBD] imaging studies collected from [TBD] between [TBD]. "
                                    "It is intended to support research in [TBD].",
                        "background": "The motivation for creating this dataset is to enable reproducible research in [TBD] "
                                      "and address limitations of existing datasets.",
                        "contents": "Included: DICOM images, a CSV with basic clinical variables, and a label file describing [TBD].",
                        "cohort": "The cohort includes subjects meeting [TBD] criteria. Exclusions include [TBD].",
                        "labels": "Labels include [TBD] and were produced by [TBD]. Definitions: [TBD].",
                    }
                )
                for s in SECTIONS:
                    k = s["key"]
                    st.session_state[f"wizard_input_{k}"] = st.session_state.wizard_answers.get(k, "")
                st.rerun()

    step = int(st.session_state.wizard_step)
    total = _current_total_steps()

    if _is_review_step(step):
        _render_review_and_prefill()
        return

    section = SECTIONS[step]
    key = section["key"]

    st.subheader(section["label"])
    if section.get("guidance"):
        st.info(section["guidance"])

    st.text_area(
        "Your draft",
        placeholder=section.get("placeholder", ""),
        height=220,
        key=f"wizard_input_{key}",
    )

    # Sync widget -> wizard_answers
    st.session_state.wizard_answers[key] = (st.session_state.get(f"wizard_input_{key}") or "").strip()
    text_val = st.session_state.wizard_answers[key]

    # AI block
    st.markdown("#### AI rewrite (optional)")
    model = st.text_input("Local model name", value="llama3.2:1b", key="wizard_model_name")

    with st.expander("Troubleshoot Ollama connection"):
        if st.button("Test connection"):
            try:
                t = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
                st.write("Status:", t.status_code)
                st.json(t.json() if t.status_code == 200 else {"body": t.text[:500]})
            except Exception as e:
                st.error(str(e))

    cols = st.columns([1, 1, 2])
    with cols[0]:
        run_ai = st.button("Get AI rewrite", use_container_width=True)
    with cols[1]:
        clear_ai = st.button("Clear AI rewrite", use_container_width=True)

    if clear_ai:
        st.session_state.wizard_feedback[key] = ""
        _close_feedback_dialog_state()
        st.rerun()

    if run_ai:
        if not (text_val or "").strip():
            st.warning("Add some text first, then run AI rewrite.")
        else:
            prompt = feedback_prompt(key, section["label"], text_val, section.get("guidance", ""))
            with st.spinner("Running local AI rewrite..."):
                out = call_ollama(model=model.strip(), prompt=prompt, temperature=0.2)
            out = _clean_model_output(key, out)

            st.session_state.wizard_feedback[key] = out
            _open_feedback_dialog(key)
            st.rerun()

    fb = (st.session_state.wizard_feedback.get(key) or "").strip()
    if fb:
        st.text_area("AI rewrite", fb, height=220)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Replace my draft with AI rewrite", use_container_width=True):
                st.session_state[f"wizard_input_{key}"] = fb
                st.session_state.wizard_answers[key] = fb
                st.rerun()
        with col2:
            if st.button("Append AI rewrite below my draft", use_container_width=True):
                combined = (st.session_state.wizard_answers[key] or "").rstrip() + "\n\n" + fb
                st.session_state[f"wizard_input_{key}"] = combined
                st.session_state.wizard_answers[key] = combined
                st.rerun()

    st.markdown("---")

    # Nav buttons
    back_disabled = step <= 0
    next_disabled = False
    if section.get("required") and not (st.session_state.wizard_answers.get(key) or "").strip():
        next_disabled = True

    nav_cols = st.columns([1, 1, 3])
    with nav_cols[0]:
        if st.button("Back", disabled=back_disabled, use_container_width=True):
            _close_feedback_dialog_state()
            st.session_state.wizard_step = max(0, step - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button("Next", disabled=next_disabled, use_container_width=True):
            _close_feedback_dialog_state()
            st.session_state.wizard_step = min(total - 1, step + 1)
            st.rerun()
    with nav_cols[2]:
        if next_disabled:
            st.caption("This step is required before continuing.")


def _open_feedback_dialog(section_key: str):
    _close_feedback_dialog_state()
    st.session_state.wizard_feedback_section_key = section_key
    st.session_state.wizard_show_feedback_dialog = True


@st.dialog("AI rewrite")
def _feedback_dialog():
    key = st.session_state.wizard_feedback_section_key
    fb = (st.session_state.wizard_feedback.get(key) or "").strip()

    if not fb:
        st.info("No AI rewrite to show.")
        if st.button("Close", use_container_width=True):
            _close_feedback_dialog_state()
            st.rerun()
        return

    st.text_area("AI rewrite", fb, height=260)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Accept", type="primary", use_container_width=True):
            st.session_state[f"wizard_input_{key}"] = fb
            st.session_state.wizard_answers[key] = fb
            _close_feedback_dialog_state()
            st.rerun()

    with c2:
        if st.button("Decline", use_container_width=True):
            _close_feedback_dialog_state()
            st.rerun()


def _render_review_and_prefill():
    st.subheader("Review & Prefill")

    answers = st.session_state.wizard_answers

    st.caption("Review your sections below. You can go Back to edit, or prefill the Submit page when ready.")
    for s in SECTIONS:
        k = s["key"]
        st.markdown(f"### {s['label']}")
        val = (answers.get(k) or "").strip()
        if val:
            st.write(val)
        else:
            st.warning("TBD")

    st.markdown("---")
    compiled = compile_for_submit(answers)
    st.markdown("### Combined text (for Submit page prefill)")
    st.text_area("Combined", compiled, height=220)

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Back", use_container_width=True):
            _close_feedback_dialog_state()
            st.session_state.wizard_step = max(0, len(SECTIONS) - 1)
            st.rerun()

    with c2:
        if st.button("Prefill Submit page", use_container_width=True):
            title = (answers.get("title") or "").strip()
            st.session_state.prefill["dataset_title"] = title
            st.session_state.prefill["dataset_description"] = compiled
            st.success("Prefill set. Switch to Submit in the sidebar to see it populated.")


if __name__ == "__main__":
    wizard_page()
