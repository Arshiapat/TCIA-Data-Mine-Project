import subprocess

MODEL = "llama3"
CICADAS_PROMPT = """You are an expert biomedical data documentation specialist preparing a dataset description
for submission to **The Cancer Imaging Archive (TCIA)**.

Your task is to generate a **comprehensive dataset description** that fully satisfies the
**CICADAS (Cancer Imaging Checklist for Data Sharing)** requirements.

Follow this exact structure and use professional **Markdown formatting** with headings and bullet points.

---

### 1. Title
- Full Title (≤110 characters)
- Short Title (<30 characters, letters/numbers/dashes only)

### 2. Abstract
Brief (≤1000 characters) overview including:
- Number of subjects
- Imaging data types (CT, MRI, PET, etc.)
- Non-imaging supporting data (e.g., demographics, treatment, outcomes)
- Potential research applications

### 3. Introduction
Explain the background, purpose, and uniqueness of this dataset.
Highlight how it benefits other researchers and what problem it helps solve.

### 4. Methods
#### 4.1 Subject Inclusion & Exclusion Criteria
Describe acquisition date range, demographic/clinical characteristics, disease details, and possible selection biases.

#### 4.2 Data Acquisition
Provide acquisition details for relevant modalities:
- CT
- MRI
- PET/CT or PET/MRI
- Ultrasound
- Histopathology
- Clinical or other supporting data
- Missing data (describe any gaps)

#### 4.3 Data Analysis
Explain:
- File format conversions
- Preprocessing steps
- Annotation/segmentation protocols
- Quality control procedures
- Automated analyses (e.g., radiomics)
- Code, scripts, and software versions used

### 5. Usage Notes
Include:
- Data organization, naming, and grouping conventions
- Training/testing splits or subsets
- File format instructions
- Recommended viewing or analysis software
- Known sources of error

### 6. External Resources
Mention any related datasets, repositories, or tools that support this dataset.

### 7. Summary Paragraph
Provide a concise closing paragraph highlighting the dataset’s scientific value and how it promotes future research.

---

Dataset information to base your description on:
{dataset_info}

Generate your response now. Use Markdown syntax for readability.
Do not include any commentary, only the final structured output.
"""
dataset_info = """250 de-identified MRI brain scans of glioblastoma patients collected from 2018–2023 at two hospitals.
Includes T1, T2, and FLAIR sequences, plus clinical demographics, treatment details, and survival outcomes.
Annotations made by 3 neuroradiologists using 3D Slicer.
"""
prompt = CICADAS_PROMPT.format(dataset_info=dataset_info)

print(f"\nRunning {MODEL}...\n")
print("=== MODEL OUTPUT (streaming) ===\n")

process = subprocess.Popen(
    ["ollama", "run", MODEL, prompt],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",          # 👈 force UTF-8
    errors="replace",          # 👈 avoid crashes on any odd bytes
    bufsize=1,
)

for line in process.stdout:
    print(line, end="", flush=True)

process.wait()
print("\n\n=== END OF OUTPUT ===")
