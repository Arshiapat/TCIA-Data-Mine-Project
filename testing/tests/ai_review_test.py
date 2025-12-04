import subprocess

MODEL = "llama3"

CICADAS_REVIEW_PROMPT = """
You are a biomedical data documentation reviewer assessing a draft description for
The Cancer Imaging Archive (TCIA) submission. Evaluate it using the CICADAS checklist.

For each section below, mark COMPLETE, PARTIAL, or MISSING and add 1–2 sentences of feedback
about what needs to be added or clarified. Do not rewrite — only review.

Sections:
1. Title
2. Abstract
3. Introduction
4. Methods (Inclusion/Exclusion, Data Acquisition, Data Analysis)
5. Usage Notes
6. External Resources
7. Summary

Dataset description to review:
---
{USER_INPUT}
---
"""

dataset_draft = """
This dataset contains 250 MRI scans of glioblastoma patients between 2018–2023.
It includes T1, T2, and FLAIR images. The purpose is to support tumor segmentation studies.
Images were acquired on 3T Siemens scanners, and annotations were done with 3D Slicer.
"""

prompt = CICADAS_REVIEW_PROMPT.replace("{USER_INPUT}", dataset_draft)

print("\nRunning review via", MODEL, "...\n")
process = subprocess.Popen(
    ["ollama", "run", MODEL, prompt],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

for line in process.stdout:
    print(line, end="", flush=True)

process.wait()
print("\n\n=== END OF REVIEW ===")
