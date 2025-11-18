import streamlit as st
import json
import requests
import shlex
import subprocess

st.set_page_config(page_title="Provide a Dataset Description", layout="centered")

OLLAMA_HOST = "http://127.0.0.1:11434"  # prefer 127.0.0.1 on Windows

# ---------------- CICADAS logic ----------------
def manual_cicadas_check(text: str):
    checks = {
        "Title": ["title", "dataset name"],
        "Abstract": ["subject", "number", "modality", "application", "abstract"],
        "Introduction": ["purpose", "background", "benefit", "introduction"],
        "Methods: Inclusion/Exclusion": ["inclusion", "exclusion", "criteria"],
        "Methods: Data Acquisition": ["acquisition", "scanner", "kvp", "te", "tr", "slice", "contrast"],
        "Methods: Data Analysis": ["preprocessing", "annotation", "segmentation", "qc", "quality", "software", "version"],
        "Usage Notes": ["organization", "naming", "subset", "split", "software", "format"],
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
You are a technical editor that enforces the CICADAS dataset description checklist.

CICADAS sections:
1) Title
2) Abstract
3) Introduction
4) Methods: Inclusion/Exclusion
5) Methods: Data Acquisition
6) Methods: Data Analysis
7) Usage Notes
8) External Resources
9) Summary

Tasks:
- Read the dataset description.
- Identify which CICADAS sections are present and which are missing.
- For each missing section, write concise action items with concrete details the author must add.
- Produce a revised draft that keeps the author's voice but fills obvious gaps and prompts for unknowns with [TBD] tags.
- Return a short JSON summary at the end with keys: present_sections, missing_sections.

Author text:
\"\"\"{description_text.strip()}\"\"\"


Current detection:
Present: {present if present else "None"}
Missing: {missing if missing else "None"}

Format:
1) Overview
2) Missing sections with bullet action items
3) Revised draft
4) JSON summary line as a single JSON object
"""

# ---------------- Ollama helpers ----------------
def ollama_probe(host=OLLAMA_HOST):
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", [])]
        return True, models
    except Exception as e:
        return False, str(e)

def call_ollama(model: str, prompt: str, temperature: float = 0.2):
    """
    Robust caller:
    1) Try /api/chat non streaming
    2) Fallback to /api/generate streaming line JSON
    3) Final fallback to CLI
    """
    # 1) /api/chat
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature},
                "stream": False
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
                "options": {"temperature": temperature},
                "stream": True
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
    except Exception as e:
        # fall through to CLI
        pass

    # 3) CLI fallback
    try:
        cmd = f'ollama run {shlex.quote(model)} --temperature {temperature}'
        result = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=180,
            shell=True,
        )
        if result.returncode != 0:
            return f"[Ollama error] {result.stderr.decode(errors='ignore').strip()}"
        return result.stdout.decode(errors='ignore').strip()
    except Exception as e:
        return f"[Ollama error] {e}"

# ---------------- Session state ----------------
ss = st.session_state
ss.setdefault("all_ok", False)
ss.setdefault("report", [])
ss.setdefault("overview_out", "")
ss.setdefault("revise_out", "")
ss.setdefault("last_prompt", "")
ss.setdefault("ollama_model", "llama3")  # default to llama3 for your feature
ss.setdefault("probe_result", None)

# ---------------- UI ----------------
st.title("Provide a Dataset Description")

# Probe card
if st.button("Test Ollama connection"):
    ok, data = ollama_probe()
    ss["probe_result"] = (ok, data)

if ss.get("probe_result"):
    ok, data = ss["probe_result"]
    if ok:
        st.success(f"Ollama OK at {OLLAMA_HOST}. Models: {', '.join(data) or 'none installed'}")
    else:
        st.error(f"Ollama not reachable: {data}")

user_text = st.text_area(
    "Paste your description",
    height=240,
    placeholder="This dataset contains ...",
)

colA, colB, colC = st.columns([1, 1, 1])
do_manual = colA.button("Manual check")
do_revise = colB.button("Revise with AI")
do_combo = colC.button("Manual check → Revise with AI")

# Run manual check when the text changes or when the user clicks Manual check
if user_text.strip() and (do_manual or "auto_checked" not in ss):
    ss["all_ok"], ss["report"] = manual_cicadas_check(user_text)
    ss["auto_checked"] = True

# Collapsible checklist stays on screen
with st.expander("CICADAS checklist details", expanded=not ss["all_ok"]):
    if not user_text.strip():
        st.info("Paste a description above to run the checklist.")
    else:
        if ss["all_ok"]:
            st.success("CICADAS checklist satisfied")
        else:
            st.error("CICADAS checklist not complete")
        for section, present, kws in ss["report"]:
            if present:
                st.success(f"{section}: detected")
            else:
                st.warning(f"{section}: missing or incomplete")
                st.caption("Consider adding: " + ", ".join(kws))
        st.caption("Use the buttons below to re-check or ask AI to revise.")

st.divider()

# Always-visible controls
st.text_input("Ollama model name", key="ollama_model", help="Examples: llama3, llama3.1, qwen2.5, mistral, or a fine-tuned name")

# Actions
def run_revise_pipeline():
    if user_text.strip():
        all_ok, report = manual_cicadas_check(user_text)
        ss["all_ok"], ss["report"] = all_ok, report
        prompt = cicadas_prompt(user_text, report)
        ss["last_prompt"] = prompt
        with st.spinner("Calling Ollama for revision..."):
            ai_text = call_ollama(ss["ollama_model"], prompt)
        ss["revise_out"] = ai_text
    else:
        st.warning("Please paste a description first.")

if do_combo:
    run_revise_pipeline()

if do_revise and not do_combo:
    if not user_text.strip():
        st.warning("Please paste a description first.")
    else:
        if "auto_checked" not in ss:
            ss["all_ok"], ss["report"] = manual_cicadas_check(user_text)
            ss["auto_checked"] = True
        prompt = cicadas_prompt(user_text, ss["report"])
        ss["last_prompt"] = prompt
        with st.spinner("Calling Ollama for revision..."):
            ai_text = call_ollama(ss["ollama_model"], prompt)
        ss["revise_out"] = ai_text

# Output panes
if ss.get("last_prompt"):
    with st.expander("Prompt sent to the model"):
        st.code(ss["last_prompt"])

if ss.get("revise_out"):
    st.subheader("AI revision")
    st.text_area("Revised draft", ss["revise_out"], height=360)

# Optional secondary action: overview-only without rewriting
if st.button("Run AI overview only"):
    if not user_text.strip():
        st.warning("Please paste a description first.")
    else:
        if "auto_checked" not in ss:
            ss["all_ok"], ss["report"] = manual_cicadas_check(user_text)
            ss["auto_checked"] = True
        prompt = cicadas_prompt(user_text, ss["report"])
        ss["last_prompt"] = prompt
        with st.spinner("Calling Ollama for overview..."):
            ai_text = call_ollama(ss["ollama_model"], prompt)
        ss["overview_out"] = ai_text
        with st.expander("AI overview"):
            st.text(ss["overview_out"])
