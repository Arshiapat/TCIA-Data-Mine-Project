#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

# 1. Download Ollama on https://ollama.com
# 2. Download the model(in this sample: mistral)
# 2. This is a small sample to run mistral model in ollama.

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_generate(prompt, model="mistral"):
    """Call Ollama API and correctly parse streaming JSON responses."""
    payload = {"model": model, "prompt": prompt, "stream": True}
    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    if response.status_code != 200:
        print(f"Ollama request failed: {response.status_code}")
        return ""

    full_text = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if "response" in data:
                full_text += data["response"]
        except json.JSONDecodeError:
            continue

    return full_text.strip()

# === LLM-Generated Sections ===
# You can change the prompt.
uniq_prompt = f"""
You are writing the "Uniqueness to TCIA" section for a TCIA dataset slide.
Summarize in 2–3 short sentences what makes this dataset unique compared to existing TCIA collections
"""

uniq_summary = ollama_generate(uniq_prompt)
print(uniq_summary)

    
    