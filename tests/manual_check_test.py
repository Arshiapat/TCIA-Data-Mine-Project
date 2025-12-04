def check_cicadas_completeness(dataset_text: str):
    checks = {
        "Title": ["title", "dataset name"],
        "Abstract": ["subject", "number", "modality", "application"],
        "Introduction": ["purpose", "background", "benefit"],
        "Methods": ["criteria", "acquisition", "analysis"],
        "Usage Notes": ["organization", "naming", "subset", "software"],
        "External Resources": ["repository", "dataset", "code", "tool"],
        "Summary": ["summary", "conclusion", "value"]
    }

    print("\n=== MANUAL CHECK REPORT ===\n")

    for section, keywords in checks.items():
        found = any(word.lower() in dataset_text.lower() for word in keywords)
        if found:
            print(f"✅ {section} — appears present (based on keyword match).")
        else:
            print(f"⚠️ {section} — possibly missing or incomplete.")
            print(f"   ⤷ Consider adding details like: {', '.join(keywords)}\n")


# Example test input
dataset_draft = """
This dataset contains 250 MRI scans of glioblastoma patients between 2018–2023.
It includes T1, T2, and FLAIR images. The purpose is to support tumor segmentation studies.
Images were acquired on 3T Siemens scanners, and annotations were done with 3D Slicer.
"""

check_cicadas_completeness(dataset_draft)
