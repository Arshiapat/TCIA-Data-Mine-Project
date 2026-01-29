import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any
import pandas as pd
import numpy as np

# ============================================================
# Data Inventory & Standardization: Medical Imaging (DICOM)
# ============================================================

class DataInventoryStandardizer:
    def __init__(self):
        self.inventory = {}

    # -------------------------------
    # Public API
    # -------------------------------

    def analyze_directory(self, root_path: str):
        root_path = Path(root_path)
        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_path}")

        print("Starting data inventory...")

        self.inventory = {
            "dataset_overview": self._dataset_overview(root_path),
            "file_format_inventory": self._file_format_inventory(root_path),
            "dicom_summary": self._dicom_summary(root_path)
        }

        # Standardized DICOM output
        self.extract_dicom_series_metadata(
            root_path,
            output_tsv="dicom_series_inventory.tsv"
        )

        return self.inventory

    # -------------------------------
    # Dataset inventory
    # -------------------------------

    def _dataset_overview(self, root_path: Path) -> Dict[str, Any]:
        total_size = 0
        total_files = 0

        for f in root_path.rglob("*"):
            if f.is_file():
                total_files += 1
                total_size += f.stat().st_size

        return {
            "total_files": total_files,
            "total_size_gb": round(total_size / (1024 ** 3), 2),
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }

    def _file_format_inventory(self, root_path: Path) -> Dict[str, Any]:
        formats = defaultdict(lambda: {"count": 0, "size_bytes": 0})

        for f in root_path.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower() if f.suffix else "no_extension"
                formats[ext]["count"] += 1
                formats[ext]["size_bytes"] += f.stat().st_size

        return {
            ext: {
                "file_count": info["count"],
                "total_size_gb": round(info["size_bytes"] / (1024 ** 3), 4)
            }
            for ext, info in formats.items()
        }

    # -------------------------------
    # DICOM inventory (high level)
    # -------------------------------

    def _dicom_summary(self, root_path: Path) -> Dict[str, Any]:
        try:
            import pydicom
        except ImportError:
            return {"error": "pydicom not installed"}

        dicom_files = list(root_path.rglob("*.dcm")) + list(root_path.rglob("*.dicom"))

        return {
            "dicom_file_count": len(dicom_files)
        }

    # -------------------------------
    # STANDARDIZED OUTPUT
    # -------------------------------

    def extract_dicom_series_metadata(self, root_path: Path, output_tsv: str):
        """
        Extract standardized series-level metadata.
        One row per SeriesInstanceUID.
        """
        try:
            import pydicom
            from pydicom.errors import InvalidDicomError
        except ImportError:
            raise RuntimeError("pydicom is required")

        dicom_files = list(root_path.rglob("*.dcm")) + list(root_path.rglob("*.dicom"))
        if not dicom_files:
            print("No DICOM files found.")
            return

        print(f"Extracting metadata from {len(dicom_files)} DICOM files...")

        series_index = {}

        for i, path in enumerate(dicom_files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(dicom_files)}")

            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)

                series_uid = self._safe_get(ds, "SeriesInstanceUID")
                if not series_uid:
                    continue

                if series_uid not in series_index:
                    series_index[series_uid] = {
                        "PatientID": self._safe_get(ds, "PatientID"),
                        "StudyInstanceUID": self._safe_get(ds, "StudyInstanceUID"),
                        "StudyDate": self._safe_get(ds, "StudyDate"),
                        "StudyDescription": self._safe_get(ds, "StudyDescription"),
                        "SeriesInstanceUID": series_uid,
                        "SeriesDescription": self._safe_get(ds, "SeriesDescription"),
                        "Manufacturer": self._safe_get(ds, "Manufacturer"),
                        "ManufacturerModelName": self._safe_get(ds, "ManufacturerModelName"),
                        "Modality": self._safe_get(ds, "Modality"),
                        "BodyPartExamined": self._safe_get(ds, "BodyPartExamined")
                    }

            except (InvalidDicomError, Exception):
                continue

        df = pd.DataFrame(series_index.values())

        df.sort_values(
            by=["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
            inplace=True,
            ignore_index=True
        )

        df.to_csv(output_tsv, sep="\t", index=False)
        print(f"✔ Series-level TSV written to {output_tsv}")

    # -------------------------------
    # Helpers
    # -------------------------------

    def _safe_get(self, ds, attr: str) -> str:
        return str(getattr(ds, attr, "")).strip()

    # -------------------------------
    # Save inventory
    # -------------------------------

    def save_inventory(self, output_json="data_inventory.json"):
        with open(output_json, "w") as f:
            json.dump(self.inventory, f, indent=2)
        print(f"✔ Inventory JSON written to {output_json}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    analyzer = DataInventoryStandardizer()

    folder = input("Enter path to data directory (blank = current): ").strip()
    if not folder:
        folder = "."

    inventory = analyzer.analyze_directory(folder)
    analyzer.save_inventory()

    print("\nData Inventory & Standardization complete.")

