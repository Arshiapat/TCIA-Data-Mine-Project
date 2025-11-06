import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class MedicalDataAnalyzer:
    def __init__(self):
        self.results = {}
    
    def analyze_directory(self, root_path: str) -> Dict[str, Any]:
        """Main analysis function - analyzes EVERY file in the directory"""
        root_path = Path(root_path)
        if not root_path.exists():
            raise ValueError(f"Directory {root_path} does not exist")
        
        print("Scanning directory structure...")
        self.results = {
            "dataset_overview": self._get_dataset_overview(root_path),
            "all_file_formats": self._analyze_all_file_formats(root_path),
            "medical_imaging_summary": self._analyze_medical_images(root_path),
            "tabular_data_summary": self._analyze_tabular_data(root_path)
        }
        
        return self.results
    
    def _convert_to_native(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_native(item) for item in obj)
        else:
            return obj
    
    def _get_dataset_overview(self, root_path: Path) -> Dict[str, Any]:
        """Calculate basic dataset statistics - analyzes ALL files"""
        total_size = 0
        file_count = 0
        
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size
        
        return self._convert_to_native({
            "total_disk_size_gb": round(total_size / (1024**3), 2),
            "total_file_count": file_count,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        })
    
    def _analyze_all_file_formats(self, root_path: Path) -> Dict[str, Any]:
        """Comprehensive analysis of ALL file formats in the directory"""
        print("Analyzing all file formats...")
        format_breakdown = defaultdict(lambda: {"count": 0, "total_size_bytes": 0})
        
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                file_extension = file_path.suffix.lower()
                
                # Handle no-extension files
                if not file_extension:
                    file_extension = "no_extension"
                
                format_breakdown[file_extension]["count"] += 1
                format_breakdown[file_extension]["total_size_bytes"] += file_size
        
        # Convert to regular dict and calculate GB
        result = {}
        for ext, data in format_breakdown.items():
            result[ext] = {
                "file_count": data["count"],
                "total_size_gb": round(data["total_size_bytes"] / (1024**3), 4)
            }
        
        return self._convert_to_native(result)
    
    def _analyze_medical_images(self, root_path: Path) -> Dict[str, Any]:
        """Analyze all medical image formats - processes ALL files"""
        print("Analyzing medical images...")
        image_summary = {
            "dicom": self._analyze_dicom_files(root_path),
            "nifti": self._analyze_format_files(root_path, [".nii", ".nii.gz"]),
            "histopathology": self._analyze_format_files(root_path, [".svs", ".ndpi", ".tif", ".tiff"]),
            "standard_images": self._analyze_format_files(root_path, [".jpg", ".jpeg", ".png", ".bmp"])
        }
        
        # Remove empty categories
        filtered_summary = {k: v for k, v in image_summary.items() if v["file_count"] > 0}
        return self._convert_to_native(filtered_summary)
    
    def _analyze_dicom_files(self, root_path: Path) -> Dict[str, Any]:
        """Analyze ALL DICOM files and extract hierarchy with better error handling"""
        try:
            import pydicom
            from pydicom.errors import InvalidDicomError
        except ImportError:
            return {"error": "pydicom not installed", "file_count": 0}
        
        dicom_files = list(root_path.rglob("*.dcm")) + list(root_path.rglob("*.dicom"))
        if not dicom_files:
            return {"file_count": 0}
        
        print(f"Analyzing {len(dicom_files)} DICOM files...")
        
        patients = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            'modality': 'Unknown', 
            'file_count': 0, 
            'total_size': 0
        })))
        
        processed_count = 0
        error_count = 0
        elements = []
        patient_ids = []

        sex_counts = defaultdict(int)
        age_values = []
        body_part_counts = defaultdict(int)

        # Process EVERY DICOM file
        for i, dicom_file in enumerate(dicom_files):
            if i % 100 == 0:  # Progress indicator for large datasets
                print(f"  Processed {i}/{len(dicom_files)} DICOM files...")
                
            try:
                # Read DICOM file without pixel data for performance
                ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)

                #print("Testing Dicom")
                '''
                for element in ds:
                    if element not in elements:
                        elements.append(element)
                '''
                
                # Extract hierarchy information with fallbacks
                patient_id = getattr(ds, 'PatientID', f'Unknown_{i}')
                study_uid = getattr(ds, 'StudyInstanceUID', f'Unknown_Study_{i}')
                series_uid = getattr(ds, 'SeriesInstanceUID', f'Unknown_Series_{i}')
                modality = getattr(ds, 'Modality', 'Unknown')
                image_type = getattr(ds, 'ImageType', 'Unknown_Image_Type')
                patient_sex = getattr(ds, 'PatientSex', 'Unknown')
                patient_age = getattr(ds, 'PatientAge', None)
                body_part = getattr(ds, 'BodyPartExamined', 'Unknown')


                if patient_id not in patient_ids:
                    patient_ids.append(patient_id)
                    # Update sex counts
                    sex_counts[patient_sex] += 1

                # Extract numeric age (handles formats like "045Y", "32", or missing)
                if patient_age:
                    try:
                        age_val = int(str(patient_age).strip().rstrip('Yy'))
                        age_values.append(age_val)
                    except ValueError:
                        pass  # Skip non-numeric ages

                # Update body part counts
                body_part_counts[body_part] += 1

                # Use StudyID as fallback if StudyInstanceUID is missing
                if study_uid.startswith('Unknown'):
                    study_id = getattr(ds, 'StudyID', None)
                    if study_id:
                        study_uid = f"StudyID_{study_id}"
                
                # Use SeriesNumber as fallback if SeriesInstanceUID is missing  
                if series_uid.startswith('Unknown'):
                    series_number = getattr(ds, 'SeriesNumber', None)
                    if series_number:
                        series_uid = f"SeriesNumber_{series_number}"
                
                series_info = patients[patient_id][study_uid][series_uid]
                series_info['file_count'] += 1
                series_info['total_size'] += dicom_file.stat().st_size
                series_info['modality'] = modality
                elements.append(image_type)
                
                processed_count += 1
                
            except (InvalidDicomError, Exception) as e:
                error_count += 1
                continue

        # Compute age range if available
        if age_values:
            age_summary = {"min_age": int(np.min(age_values)), "max_age": int(np.max(age_values))}
        else:
            age_summary = {"min_age": None, "max_age": None}

        # Convert defaultdicts to regular dicts
        sex_summary = dict(sex_counts)
        body_part_summary = dict(body_part_counts)

        print(f"  Successfully processed {processed_count}/{len(dicom_files)} DICOM files")
        #print(f"Elements: {elements}")
        if error_count > 0:
            print(f"  Failed to process {error_count} DICOM files due to errors")
        
        # Calculate summary statistics
        modality_breakdown = defaultdict(lambda: {"series_count": 0, "file_count": 0})
        total_file_count = 0
        
        for patient in patients.values():
            for study in patient.values():
                for series_uid, series_data in study.items():
                    modality = series_data['modality']
                    modality_breakdown[modality]["series_count"] += 1
                    modality_breakdown[modality]["file_count"] += series_data['file_count']
                    total_file_count += series_data['file_count']

        # Normalize and flatten all elements before deduplication
        flat_elements = []
        for el in elements:
            if isinstance(el, (list, tuple, np.ndarray)):
                flat_elements.extend(map(str, el))
            else:
                flat_elements.append(str(el))

        result = {
            "patient_count": len(patients),
            "study_count": sum(len(patient) for patient in patients.values()),
            "series_count": sum(len(study) for patient in patients.values() for study in patient.values()),
            "file_count": total_file_count,
            "modality_breakdown": dict(modality_breakdown),
            "processing_stats": {
                "total_files_found": len(dicom_files),
                "image_types": list(np.unique(np.array(flat_elements))),
                "successfully_processed": processed_count,
                "failed_to_process": error_count
            }
        }

        result.update({
            "patient_demographics": {
                "sex_distribution": sex_summary,
                "age_range": age_summary
            },
            "body_part_examined_counts": body_part_summary
        })
        
        return self._convert_to_native(result)
    
    def _analyze_format_files(self, root_path: Path, extensions: List[str]) -> Dict[str, Any]:
        """Analyze ALL files with specific extensions"""
        file_count = 0
        total_size = 0
        
        for ext in extensions:
            files = list(root_path.rglob(f"*{ext}")) + list(root_path.rglob(f"*{ext.upper()}"))
            for file_path in files:
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
        
        if file_count == 0:
            return {"file_count": 0}
        
        return self._convert_to_native({
            "file_count": file_count,
            "total_size_gb": round(total_size / (1024**3), 2)
        })
    
    def _analyze_tabular_data(self, root_path: Path) -> Dict[str, Any]:
        """Analyze ALL tabular data files"""
        print("Analyzing tabular data...")
        tabular_files = []
        for ext in [".csv", ".tsv", ".xlsx", ".xls"]:
            tabular_files.extend(list(root_path.rglob(f"*{ext}")))
        
        if not tabular_files:
            return {"file_count": 0}
        
        data_dictionary = {}
        formats_found = set()
        
        # Analyze EVERY tabular file
        for file_path in tabular_files:
            try:
                if file_path.suffix.lower() in ['.csv', '.tsv']:
                    # Try to detect delimiter for CSV/TSV
                    delimiter = ',' if file_path.suffix.lower() == '.csv' else '\t'
                    df = pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
                else:  # Excel files
                    df = pd.read_excel(file_path)
                
                formats_found.add(file_path.suffix.lower())
                
                column_analysis = {}
                for column in df.columns:
                    col_data = df[column].dropna()
                    unique_vals = col_data.unique()
                    
                    col_info = {
                        "data_type": str(df[column].dtype),
                        "non_null_count": len(col_data),
                        "null_count": int(df[column].isnull().sum()),  # Convert to native int
                        "null_percentage": round((df[column].isnull().sum() / len(df)) * 100, 2),
                        "unique_value_count": len(unique_vals),
                        "unique_values_sample": unique_vals[:10].tolist() if len(unique_vals) > 0 else []
                    }
                    
                    # Add range for numeric columns
                    if np.issubdtype(df[column].dtype, np.number):
                        col_info["value_range"] = {
                            "min": float(col_data.min()),
                            "max": float(col_data.max())
                        }
                    
                    column_analysis[column] = self._convert_to_native(col_info)
                
                data_dictionary[file_path.name] = {
                    "columns": column_analysis,
                    "total_rows": int(len(df))  # Convert to native int
                }
                
            except Exception as e:
                data_dictionary[file_path.name] = {"error": str(e)}
        
        return self._convert_to_native({
            "file_count": len(tabular_files),
            "formats": list(formats_found),
            "data_dictionary": data_dictionary
        })
    
    def save_report(self, output_path: str = "medical_data_report.json"):
        """Save analysis results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"Report saved to {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")

# Custom JSON encoder as additional safety net
class NativeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Let our _convert_to_native method handle the conversion
        return obj

# USAGE EXAMPLE
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MedicalDataAnalyzer()
    
    # Analyze your directory (replace with your path)
    folder_path = input("Enter the path to your medical data folder: ").strip()
    
    if not folder_path:
        # Demo with current directory
        folder_path = "."
        print("Using current directory for demo...")
    
    try:
        print("Analyzing ALL medical data... This may take a while for large datasets.")
        results = analyzer.analyze_directory(folder_path)
        
        # Print summary
        print("\n" + "="*50)
        print("MEDICAL DATA ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total Size: {results['dataset_overview']['total_disk_size_gb']} GB")
        print(f"Total Files: {results['dataset_overview']['total_file_count']}")
        
        # Show all file formats found
        print(f"\nALL FILE FORMATS FOUND: {len(results['all_file_formats'])} formats")
        for ext, data in sorted(results['all_file_formats'].items(), key=lambda x: x[1]['file_count'], reverse=True)[:15]:
            print(f"  {ext}: {data['file_count']} files ({data['total_size_gb']} GB)")
        
        if len(results['all_file_formats']) > 15:
            print(f"  ... and {len(results['all_file_formats']) - 15} more formats")
        
        if results['medical_imaging_summary']:
            print("\nMEDICAL IMAGES:")
            for category, data in results['medical_imaging_summary'].items():
                if 'file_count' in data and data['file_count'] > 0:
                    if category == 'dicom' and 'patient_count' in data:
                        print(f"  DICOM: {data['file_count']} files")
                        print(f"    Patients: {data['patient_count']}, Studies: {data['study_count']}, Series: {data['series_count']}")
                        if 'modality_breakdown' in data:
                            for modality, stats in data['modality_breakdown'].items():
                                print(f"    {modality}: {stats['series_count']} series, {stats['file_count']} files")
                    else:
                        print(f"  {category.upper()}: {data['file_count']} files")
        
        if results['tabular_data_summary']['file_count'] > 0:
            print(f"\nTABULAR DATA: {results['tabular_data_summary']['file_count']} files")
            for filename, fileinfo in results['tabular_data_summary']['data_dictionary'].items():
                if 'columns' in fileinfo:
                    print(f"  {filename}: {len(fileinfo['columns'])} columns")


        # Save full report
        analyzer.save_report()
        print(f"\nFull report saved to: medical_data_report.json")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
