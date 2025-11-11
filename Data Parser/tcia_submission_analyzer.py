"""
Enhanced Medical Dataset Analyzer for TCIA Submissions
Comprehensive analysis with TCIA form automation and detailed reporting
Version 4.0
"""

import os
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import mimetypes

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Initialize mimetypes
mimetypes.init()

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class TCIADatasetAnalyzer:
    """
    Enhanced medical dataset analyzer with TCIA form automation.
    """
    
    # File type classifications using multiple detection methods
    FILE_CLASSIFICATIONS = {
        'DICOM': {
            'extensions': {'.dcm', '.dicom', '.DCM', '.DICOM'},
            'magic_bytes': [b'DICM'],
            'mime_types': {'application/dicom'}
        },
        'Medical_Imaging': {
            'extensions': {'.nii', '.nii.gz', '.nrrd', '.mha', '.mhd', '.mnc', '.mgz'},
            'mime_types': set()
        },
        'Images': {
            'extensions': {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.svg', '.webp'},
            'mime_types': {'image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/gif', 'image/svg+xml'}
        },
        'Pathology_Slides': {
            'extensions': {'.svs', '.ndpi', '.scn', '.vms', '.vmu', '.mrxs', '.bif'},
            'mime_types': set()
        },
        'Tabular_Data': {
            'extensions': {'.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.feather'},
            'mime_types': {'text/csv', 'application/vnd.ms-excel', 
                          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
        },
        'Text_Documents': {
            'extensions': {'.txt', '.md', '.rst', '.log', '.readme'},
            'mime_types': {'text/plain', 'text/markdown'}
        },
        'Archives': {
            'extensions': {'.zip', '.tar', '.gz', '.7z', '.rar', '.bz2'},
            'mime_types': {'application/zip', 'application/x-tar', 'application/gzip'}
        },
        'Code_Scripts': {
            'extensions': {'.py', '.r', '.m', '.ipynb', '.sh', '.bash', '.bat'},
            'mime_types': {'text/x-python', 'text/x-script.python'}
        },
        'Documents': {
            'extensions': {'.pdf', '.doc', '.docx', '.ppt', '.pptx'},
            'mime_types': {'application/pdf', 'application/msword',
                          'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}
        },
        'Configuration': {
            'extensions': {'.json', '.xml', '.yaml', '.yml', '.ini', '.cfg', '.toml'},
            'mime_types': {'application/json', 'application/xml', 'text/xml'}
        },
        'Radiotherapy': {
            'extensions': {'.dcm'},  # Will check DICOM modality
            'modalities': {'RTPLAN', 'RTSTRUCT', 'RTDOSE', 'RTIMAGE'}
        }
    }
    
    def __init__(self, max_workers: int = 4, verbose: bool = True):
        self.max_workers = max_workers
        self.verbose = verbose
        self.results = {}
        self.processing_log = []
        self.tcia_form_data = {}
        
    def analyze_dataset(self, root_path: str) -> Dict[str, Any]:
        """Main analysis entry point."""
        root_path = Path(root_path)
        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_path}")
        
        self._log("="*80)
        self._log("TCIA DATASET ANALYZER - Starting comprehensive analysis")
        self._log("="*80)
        start_time = datetime.now()
        
        # Comprehensive analysis pipeline
        self.results = {
            "metadata": self._generate_metadata(root_path, start_time),
            "directory_structure": self._map_directory_structure(root_path),
            "dataset_overview": self._analyze_dataset_overview(root_path),
            "file_inventory": self._create_file_inventory(root_path),
            "dicom_catalog": self._catalog_dicom_comprehensive(root_path),
            "tabular_catalog": self._catalog_tabular_comprehensive(root_path),
            "image_catalog": self._catalog_images(root_path),
            "text_files_catalog": self._catalog_text_files(root_path),
            "archive_files": self._catalog_archives(root_path),
            "relationships": self._analyze_relationships(),
            "statistics_summary": {},
            "processing_log": self.processing_log
        }
        
        # Generate statistics
        self.results["statistics_summary"] = self._generate_statistics()
        
        # Generate TCIA form helper data
        self.tcia_form_data = self._generate_tcia_form_helper()
        self.results["tcia_form_helper"] = self.tcia_form_data
        
        duration = (datetime.now() - start_time).total_seconds()
        self.results["metadata"]["analysis_duration_seconds"] = round(duration, 2)
        
        self._log(f"\nAnalysis complete in {duration:.2f} seconds")
        return self.results
    
    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def _categorize_file_robust(self, file_path: Path) -> str:
        """
        Robust file categorization using multiple detection methods:
        1. Extension matching
        2. MIME type detection
        3. Magic bytes checking
        4. Content analysis for special cases
        """
        # Get basic file info
        ext = file_path.suffix.lower()
        
        # Special case: compressed medical imaging
        if file_path.name.endswith('.nii.gz'):
            return 'Medical_Imaging'
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Check each category
        for category, rules in self.FILE_CLASSIFICATIONS.items():
            # Check extension
            if ext in rules.get('extensions', set()):
                # Special handling for DICOM files
                if category == 'DICOM':
                    # Verify it's actually DICOM by checking magic bytes
                    if self._is_dicom_file(file_path):
                        return 'DICOM'
                else:
                    return category
            
            # Check MIME type
            if mime_type and mime_type in rules.get('mime_types', set()):
                return category
        
        # Check for files without extension - could be DICOM
        if not ext:
            if self._is_dicom_file(file_path):
                return 'DICOM'
            return 'No_Extension'
        
        return 'Other'
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """Check if file is DICOM by reading magic bytes."""
        try:
            with open(file_path, 'rb') as f:
                # DICOM files have 'DICM' at byte 128
                f.seek(128)
                return f.read(4) == b'DICM'
        except:
            return False
    
    def _generate_metadata(self, root_path: Path, start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive metadata."""
        return {
            "dataset_path": str(root_path.absolute()),
            "dataset_name": root_path.name,
            "analysis_timestamp": start_time.isoformat(),
            "analyzer_version": "4.0.0",
            "python_environment": {
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "python_version": sys.version
            }
        }
    
    def _map_directory_structure(self, root_path: Path) -> Dict[str, Any]:
        """Create a complete map of the directory structure."""
        self._log("\n📁 Mapping directory structure...")
        
        structure = {
            "root": str(root_path.name),
            "total_depth": 0,
            "directories_by_level": defaultdict(list),
            "directory_tree": []
        }
        
        all_dirs = []
        for item in root_path.rglob('*'):
            if item.is_dir():
                depth = len(item.relative_to(root_path).parts)
                rel_path = str(item.relative_to(root_path))
                structure["directories_by_level"][depth].append(rel_path)
                structure["total_depth"] = max(structure["total_depth"], depth)
                all_dirs.append({
                    "path": rel_path,
                    "depth": depth,
                    "name": item.name
                })
        
        structure["directories_by_level"] = {
            k: v for k, v in structure["directories_by_level"].items()
        }
        structure["total_directories"] = len(all_dirs)
        structure["directory_tree"] = all_dirs
        
        return structure
    
    def _analyze_dataset_overview(self, root_path: Path) -> Dict[str, Any]:
        """Calculate exhaustive dataset statistics."""
        self._log("📊 Analyzing dataset overview...")
        
        total_size = 0
        file_count = 0
        files_by_size = {
            "<1KB": 0, "1KB-10KB": 0, "10KB-100KB": 0, 
            "100KB-1MB": 0, "1MB-10MB": 0, "10MB-100MB": 0, 
            "100MB-1GB": 0, ">1GB": 0
        }
        
        for item in root_path.rglob('*'):
            if item.is_file():
                size = item.stat().st_size
                file_count += 1
                total_size += size
                
                # Categorize by size
                if size < 1024:
                    files_by_size["<1KB"] += 1
                elif size < 10 * 1024:
                    files_by_size["1KB-10KB"] += 1
                elif size < 100 * 1024:
                    files_by_size["10KB-100KB"] += 1
                elif size < 1024 * 1024:
                    files_by_size["100KB-1MB"] += 1
                elif size < 10 * 1024 * 1024:
                    files_by_size["1MB-10MB"] += 1
                elif size < 100 * 1024 * 1024:
                    files_by_size["10MB-100MB"] += 1
                elif size < 1024 * 1024 * 1024:
                    files_by_size["100MB-1GB"] += 1
                else:
                    files_by_size[">1GB"] += 1
        
        return {
            "total_files": file_count,
            "total_size_bytes": total_size,
            "total_size_kb": round(total_size / 1024, 2),
            "total_size_mb": round(total_size / (1024**2), 2),
            "total_size_gb": round(total_size / (1024**3), 4),
            "average_file_size_bytes": round(total_size / file_count) if file_count > 0 else 0,
            "average_file_size_kb": round((total_size / file_count) / 1024, 2) if file_count > 0 else 0,
            "average_file_size_mb": round((total_size / file_count) / (1024**2), 4) if file_count > 0 else 0,
            "files_by_size_category": files_by_size
        }
    
    def _create_file_inventory(self, root_path: Path) -> Dict[str, Any]:
        """Create comprehensive file format inventory using robust detection."""
        self._log("🔍 Creating comprehensive file inventory...")
        
        format_inventory = defaultdict(lambda: {
            "count": 0,
            "total_size": 0,
            "files": [],
            "extensions_found": set()
        })
        
        extension_stats = defaultdict(int)
        all_extensions = set()
        
        file_list = list(root_path.rglob('*'))
        if HAS_TQDM and self.verbose:
            file_iterator = tqdm(file_list, desc="Inventorying files")
        else:
            file_iterator = file_list
        
        for file_path in file_iterator:
            if file_path.is_file():
                ext = file_path.suffix.lower() or "no_extension"
                size = file_path.stat().st_size
                
                all_extensions.add(ext)
                extension_stats[ext] += 1
                
                # Robust categorization
                category = self._categorize_file_robust(file_path)
                format_inventory[category]["count"] += 1
                format_inventory[category]["total_size"] += size
                format_inventory[category]["extensions_found"].add(ext)
                
                # Store sample paths (limit to 20 per category)
                if len(format_inventory[category]["files"]) < 20:
                    format_inventory[category]["files"].append({
                        "path": str(file_path.relative_to(root_path)),
                        "size_bytes": size,
                        "size_mb": round(size / (1024**2), 4),
                        "extension": ext
                    })
        
        # Convert sets to lists for JSON serialization
        result = {}
        for category, data in format_inventory.items():
            result[category] = {
                "file_count": data["count"],
                "total_size_bytes": data["total_size"],
                "total_size_mb": round(data["total_size"] / (1024**2), 2),
                "total_size_gb": round(data["total_size"] / (1024**3), 4),
                "percentage_of_total": 0,  # Will calculate later
                "extensions": sorted(list(data["extensions_found"])),
                "sample_files": data["files"]
            }
        
        # Calculate percentages
        total_files = sum(cat["file_count"] for cat in result.values())
        for category in result:
            result[category]["percentage_of_total"] = round(
                (result[category]["file_count"] / total_files * 100), 2
            ) if total_files > 0 else 0
        
        return {
            "by_category": result,
            "all_extensions_found": sorted(list(all_extensions)),
            "extension_counts": dict(sorted(extension_stats.items(), 
                                           key=lambda x: x[1], reverse=True)),
            "total_categories": len(result),
            "total_unique_extensions": len(all_extensions)
        }
    
    def _catalog_dicom_comprehensive(self, root_path: Path) -> Dict[str, Any]:
        """Exhaustive DICOM cataloging with all available metadata."""
        self._log("\n🏥 Cataloging DICOM files comprehensively...")
        
        try:
            import pydicom
            from pydicom.errors import InvalidDicomError
        except ImportError:
            self._log("⚠️  pydicom not installed - DICOM cataloging skipped")
            return {"status": "pydicom_not_available"}
        
        # Find all DICOM files using robust detection
        dicom_files = []
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                if self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
        
        if not dicom_files:
            self._log("ℹ️  No DICOM files found")
            return {"file_count": 0, "status": "no_dicom_files_found"}
        
        self._log(f"Found {len(dicom_files)} DICOM files - extracting metadata...")
        
        # Comprehensive data structures
        patients = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        all_tags_found = set()
        modalities = defaultdict(int)
        sop_classes = defaultdict(int)
        transfer_syntaxes = defaultdict(int)
        manufacturers = defaultdict(int)
        manufacturer_models = defaultdict(int)
        body_parts = defaultdict(int)
        study_descriptions = defaultdict(int)
        series_descriptions = defaultdict(int)
        image_orientations = []
        pixel_spacings = []
        slice_thicknesses = []
        
        # Radiotherapy specific
        rt_structures = defaultdict(int)
        
        processed = 0
        failed = 0
        dicom_details = []
        
        if HAS_TQDM and self.verbose:
            dicom_iterator = tqdm(dicom_files, desc="Processing DICOM")
        else:
            dicom_iterator = dicom_files
        
        for dicom_path in dicom_iterator:
            try:
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
                
                # Extract ALL available tags
                tags_in_file = set()
                file_metadata = {}
                
                for elem in ds:
                    tag_name = elem.name
                    tags_in_file.add(tag_name)
                    all_tags_found.add(tag_name)
                    
                    if elem.VR != 'SQ':
                        try:
                            file_metadata[tag_name] = str(elem.value)
                        except:
                            file_metadata[tag_name] = "<unable to convert>"
                
                # Core identifiers
                patient_id = str(getattr(ds, 'PatientID', 'Unknown'))
                study_uid = str(getattr(ds, 'StudyInstanceUID', 'Unknown'))
                series_uid = str(getattr(ds, 'SeriesInstanceUID', 'Unknown'))
                sop_uid = str(getattr(ds, 'SOPInstanceUID', 'Unknown'))
                
                # Clinical metadata
                modality = str(getattr(ds, 'Modality', 'Unknown'))
                sop_class = str(getattr(ds, 'SOPClassUID', 'Unknown'))
                transfer_syntax = str(getattr(ds, 'file_meta', {}).get('TransferSyntaxUID', 'Unknown'))
                manufacturer = str(getattr(ds, 'Manufacturer', 'Unknown'))
                model = str(getattr(ds, 'ManufacturerModelName', 'Unknown'))
                body_part = str(getattr(ds, 'BodyPartExamined', 'Unknown'))
                study_desc = str(getattr(ds, 'StudyDescription', 'Unknown'))
                series_desc = str(getattr(ds, 'SeriesDescription', 'Unknown'))
                
                # Update statistics
                modalities[modality] += 1
                sop_classes[sop_class] += 1
                transfer_syntaxes[transfer_syntax] += 1
                if manufacturer != 'Unknown':
                    manufacturers[manufacturer] += 1
                if model != 'Unknown':
                    manufacturer_models[model] += 1
                if body_part != 'Unknown':
                    body_parts[body_part] += 1
                if study_desc != 'Unknown':
                    study_descriptions[study_desc] += 1
                if series_desc != 'Unknown':
                    series_descriptions[series_desc] += 1
                
                # Radiotherapy tracking
                if modality in {'RTPLAN', 'RTSTRUCT', 'RTDOSE', 'RTIMAGE'}:
                    rt_structures[modality] += 1
                
                # Technical parameters
                if hasattr(ds, 'ImageOrientationPatient'):
                    image_orientations.append(list(ds.ImageOrientationPatient))
                if hasattr(ds, 'PixelSpacing'):
                    pixel_spacings.append(list(ds.PixelSpacing))
                if hasattr(ds, 'SliceThickness'):
                    slice_thicknesses.append(float(ds.SliceThickness))
                
                # Build hierarchy
                if series_uid not in patients[patient_id][study_uid]:
                    patients[patient_id][study_uid][series_uid] = {
                        'modality': modality,
                        'series_description': series_desc,
                        'series_number': str(getattr(ds, 'SeriesNumber', 'Unknown')),
                        'instances': [],
                        'instance_count': 0,
                        'total_size': 0
                    }
                
                patients[patient_id][study_uid][series_uid]['instances'].append(sop_uid)
                patients[patient_id][study_uid][series_uid]['instance_count'] += 1
                patients[patient_id][study_uid][series_uid]['total_size'] += dicom_path.stat().st_size
                
                # Store detailed record (limit to first 100)
                if len(dicom_details) < 100:
                    dicom_details.append({
                        "file_path": str(dicom_path.relative_to(root_path)),
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "series_uid": series_uid,
                        "sop_instance_uid": sop_uid,
                        "modality": modality,
                        "body_part": body_part,
                        "tags_present": len(tags_in_file)
                    })
                
                processed += 1
                
            except Exception as e:
                failed += 1
                if failed <= 10:  # Log first 10 failures
                    self._log(f"Failed to process {dicom_path.name}: {str(e)}")
        
        # Calculate comprehensive statistics
        total_patients = len(patients)
        total_studies = sum(len(studies) for studies in patients.values())
        total_series = sum(
            len(series) 
            for studies in patients.values() 
            for series in studies.values()
        )
        total_instances = sum(
            series_data['instance_count']
            for studies in patients.values()
            for study in studies.values()
            for series_data in study.values()
        )
        
        # Patient hierarchy summary
        patient_summary = []
        for patient_id, studies in list(patients.items())[:100]:  # First 100 patients
            patient_info = {
                "patient_id": patient_id,
                "study_count": len(studies),
                "series_count": sum(len(series) for series in studies.values()),
                "total_instances": sum(
                    series_data['instance_count']
                    for study in studies.values()
                    for series_data in study.values()
                ),
                "modalities": list(set(
                    series_data['modality']
                    for study in studies.values()
                    for series_data in study.values()
                ))
            }
            patient_summary.append(patient_info)
        
        results = {
            "total_files_found": len(dicom_files),
            "successfully_processed": processed,
            "failed_to_process": failed,
            "processing_success_rate": round((processed / len(dicom_files)) * 100, 2) if len(dicom_files) > 0 else 0,
            
            "hierarchy": {
                "patient_count": total_patients,
                "study_count": total_studies,
                "series_count": total_series,
                "instance_count": total_instances,
                "avg_studies_per_patient": round(total_studies / total_patients, 2) if total_patients > 0 else 0,
                "avg_series_per_study": round(total_series / total_studies, 2) if total_studies > 0 else 0,
                "avg_instances_per_series": round(total_instances / total_series, 2) if total_series > 0 else 0
            },
            
            "modalities_found": dict(sorted(modalities.items(), key=lambda x: x[1], reverse=True)),
            "sop_classes_found": dict(sorted(sop_classes.items(), key=lambda x: x[1], reverse=True)),
            "transfer_syntaxes_found": dict(sorted(transfer_syntaxes.items(), key=lambda x: x[1], reverse=True)),
            "manufacturers_found": dict(sorted(manufacturers.items(), key=lambda x: x[1], reverse=True)),
            "manufacturer_models_found": dict(sorted(manufacturer_models.items(), key=lambda x: x[1], reverse=True)),
            "body_parts_examined": dict(sorted(body_parts.items(), key=lambda x: x[1], reverse=True)),
            "study_descriptions_found": dict(sorted(study_descriptions.items(), key=lambda x: x[1], reverse=True)),
            "series_descriptions_found": dict(sorted(series_descriptions.items(), key=lambda x: x[1], reverse=True)[:50]),
            
            "radiotherapy": {
                "has_rt_data": len(rt_structures) > 0,
                "rt_structure_counts": dict(rt_structures)
            },
            
            "technical_parameters": {
                "unique_image_orientations": len(set(tuple(x) for x in image_orientations)),
                "unique_pixel_spacings": len(set(tuple(x) for x in pixel_spacings)),
                "slice_thickness_stats": {
                    "count": len(slice_thicknesses),
                    "min": float(np.min(slice_thicknesses)) if slice_thicknesses else None,
                    "max": float(np.max(slice_thicknesses)) if slice_thicknesses else None,
                    "mean": float(np.mean(slice_thicknesses)) if slice_thicknesses else None,
                    "std": float(np.std(slice_thicknesses)) if slice_thicknesses else None
                }
            },
            
            "dicom_tags": {
                "total_unique_tags_found": len(all_tags_found),
                "all_tags": sorted(list(all_tags_found))[:100]  # First 100 tags
            },
            
            "patient_details": patient_summary,
            "sample_dicom_records": dicom_details
        }
        
        return results
    
    def _catalog_tabular_comprehensive(self, root_path: Path) -> Dict[str, Any]:
        """Comprehensive tabular data cataloging."""
        self._log("\n📋 Cataloging tabular data...")
        
        tabular_files = []
        for ext in ['.csv', '.tsv', '.xlsx', '.xls', '.parquet']:
            tabular_files.extend(list(root_path.rglob(f'*{ext}')))
        
        if not tabular_files:
            self._log("ℹ️  No tabular files found")
            return {"file_count": 0, "status": "no_tabular_files_found"}
        
        self._log(f"Found {len(tabular_files)} tabular files")
        
        file_catalog = {}
        total_rows = 0
        total_columns = 0
        all_column_names = set()
        column_type_dist = defaultdict(int)
        patient_id_columns = []
        
        for file_path in tabular_files:
            try:
                # Read file
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path, low_memory=False)
                elif file_path.suffix.lower() == '.tsv':
                    df = pd.read_csv(file_path, sep='\t', low_memory=False)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    continue
                
                row_count = len(df)
                total_rows += row_count
                total_columns += len(df.columns)
                
                # Column analysis
                column_profiles = {}
                for col in df.columns:
                    all_column_names.add(col)
                    dtype = str(df[col].dtype)
                    column_type_dist[dtype] += 1
                    
                    # Check if this might be a patient ID column
                    if any(term in col.lower() for term in ['patient', 'subject', 'participant', 'id']):
                        patient_id_columns.append({
                            "file": file_path.name,
                            "column": col,
                            "unique_count": int(df[col].nunique())
                        })
                    
                    non_null = df[col].dropna()
                    unique_vals = non_null.unique()
                    
                    profile = {
                        "dtype": dtype,
                        "total_count": len(df[col]),
                        "non_null_count": int(df[col].notna().sum()),
                        "null_count": int(df[col].isna().sum()),
                        "null_percentage": round((df[col].isna().sum() / len(df)) * 100, 2),
                        "unique_count": int(df[col].nunique()),
                        "uniqueness_ratio": round(df[col].nunique() / len(df), 4) if len(df) > 0 else 0
                    }
                    
                    # Type-specific stats
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if len(non_null) > 0:
                            profile["numeric_stats"] = {
                                "min": float(non_null.min()),
                                "max": float(non_null.max()),
                                "mean": float(non_null.mean()),
                                "median": float(non_null.median()),
                                "std": float(non_null.std())
                            }
                    elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                        if len(non_null) > 0:
                            value_counts = non_null.value_counts()
                            profile["categorical_stats"] = {
                                "top_10_values": {str(k): int(v) for k, v in value_counts.head(10).items()},
                                "avg_string_length": float(non_null.astype(str).str.len().mean()),
                                "min_string_length": int(non_null.astype(str).str.len().min()),
                                "max_string_length": int(non_null.astype(str).str.len().max())
                            }
                            profile["sample_values"] = [str(v) for v in unique_vals[:20]]
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        if len(non_null) > 0:
                            profile["datetime_stats"] = {
                                "min_date": str(non_null.min()),
                                "max_date": str(non_null.max()),
                                "date_range_days": (pd.to_datetime(non_null.max()) - pd.to_datetime(non_null.min())).days
                            }
                    
                    column_profiles[col] = profile
                
                # File summary
                file_catalog[file_path.name] = {
                    "file_path": str(file_path.relative_to(root_path)),
                    "file_size_bytes": file_path.stat().st_size,
                    "file_size_mb": round(file_path.stat().st_size / (1024**2), 2),
                    "row_count": row_count,
                    "column_count": len(df.columns),
                    "column_names": list(df.columns),
                    "columns": column_profiles,
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
                    "duplicate_row_count": int(df.duplicated().sum()),
                    "duplicate_row_percentage": round((df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 0
                }
                
            except Exception as e:
                file_catalog[file_path.name] = {
                    "file_path": str(file_path.relative_to(root_path)),
                    "error": str(e),
                    "status": "failed_to_parse"
                }
                self._log(f"Error parsing {file_path.name}: {str(e)}")
        
        return {
            "file_count": len(tabular_files),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "unique_column_names": sorted(list(all_column_names)),
            "column_type_distribution": dict(sorted(column_type_dist.items(), key=lambda x: x[1], reverse=True)),
            "patient_id_columns": patient_id_columns,
            "files": file_catalog
        }
    
    def _catalog_images(self, root_path: Path) -> Dict[str, Any]:
        """Catalog standard image files."""
        self._log("\n🖼️  Cataloging image files...")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.svg', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(root_path.rglob(f'*{ext}')))
            image_files.extend(list(root_path.rglob(f'*{ext.upper()}')))
        
        image_files = list(set(image_files))
        
        if not image_files:
            return {"file_count": 0, "status": "no_image_files_found"}
        
        format_breakdown = defaultdict(lambda: {"count": 0, "total_size": 0, "files": []})
        
        for img_file in image_files:
            ext = img_file.suffix.lower()
            size = img_file.stat().st_size
            format_breakdown[ext]["count"] += 1
            format_breakdown[ext]["total_size"] += size
            if len(format_breakdown[ext]["files"]) < 10:
                format_breakdown[ext]["files"].append(str(img_file.relative_to(root_path)))
        
        result = {}
        for ext, data in format_breakdown.items():
            result[ext] = {
                "file_count": data["count"],
                "total_size_mb": round(data["total_size"] / (1024**2), 2),
                "average_size_kb": round((data["total_size"] / data["count"]) / 1024, 2),
                "sample_files": data["files"]
            }
        
        return {
            "file_count": len(image_files),
            "total_size_mb": round(sum(f.stat().st_size for f in image_files) / (1024**2), 2),
            "formats": result
        }
    
    def _catalog_text_files(self, root_path: Path) -> Dict[str, Any]:
        """Catalog all text-based files."""
        self._log("📄 Cataloging text files...")
        
        text_extensions = ['.txt', '.md', '.rst', '.log', '.readme', '.license']
        text_files = []
        
        for ext in text_extensions:
            text_files.extend(list(root_path.rglob(f'*{ext}')))
        
        text_files = list(set(text_files))
        
        if not text_files:
            return {"file_count": 0, "status": "no_text_files_found"}
        
        text_catalog = []
        total_lines = 0
        
        for text_file in text_files[:50]:
            try:
                with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    total_lines += len(lines)
                    
                    text_catalog.append({
                        "file_path": str(text_file.relative_to(root_path)),
                        "file_name": text_file.name,
                        "extension": text_file.suffix or "no_extension",
                        "size_bytes": text_file.stat().st_size,
                        "line_count": len(lines),
                        "character_count": len(content),
                        "preview": content[:500]
                    })
            except Exception as e:
                text_catalog.append({
                    "file_path": str(text_file.relative_to(root_path)),
                    "error": str(e)
                })
        
        return {
            "file_count": len(text_files),
            "total_lines": total_lines,
            "files": text_catalog
        }
    
    def _catalog_archives(self, root_path: Path) -> Dict[str, Any]:
        """Catalog archive files."""
        self._log("📦 Cataloging archive files...")
        
        archive_extensions = ['.zip', '.tar', '.gz', '.7z', '.rar', '.bz2']
        archive_files = []
        
        for ext in archive_extensions:
            archive_files.extend(list(root_path.rglob(f'*{ext}')))
        
        if not archive_files:
            return {"file_count": 0, "status": "no_archive_files_found"}
        
        archives = []
        for archive in archive_files:
            archives.append({
                "file_path": str(archive.relative_to(root_path)),
                "file_name": archive.name,
                "extension": archive.suffix,
                "size_mb": round(archive.stat().st_size / (1024**2), 2)
            })
        
        return {
            "file_count": len(archive_files),
            "total_size_mb": round(sum(a.stat().st_size for a in archive_files) / (1024**2), 2),
            "files": archives
        }
    
    def _analyze_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between different data types."""
        self._log("🔗 Analyzing data relationships...")
        
        relationships = {}
        
        dicom_data = self.results.get("dicom_catalog", {})
        tabular_data = self.results.get("tabular_catalog", {})
        
        if dicom_data.get("hierarchy", {}).get("patient_count", 0) > 0 and tabular_data.get("file_count", 0) > 0:
            dicom_patients = set()
            if "patient_details" in dicom_data:
                dicom_patients = set(p["patient_id"] for p in dicom_data["patient_details"])
            
            csv_patients = set()
            for file_name, file_data in tabular_data.get("files", {}).items():
                if "columns" in file_data:
                    for col_name, col_data in file_data["columns"].items():
                        if any(term in col_name.lower() for term in ['patient', 'subject', 'participant']):
                            if "sample_values" in col_data:
                                csv_patients.update(col_data["sample_values"])
            
            if dicom_patients and csv_patients:
                common = dicom_patients & csv_patients
                relationships["patient_overlap"] = {
                    "dicom_patient_count": len(dicom_patients),
                    "csv_patient_count": len(csv_patients),
                    "common_patients": len(common),
                    "dicom_only": len(dicom_patients - csv_patients),
                    "csv_only": len(csv_patients - dicom_patients)
                }
        
        return relationships
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics summary."""
        stats = {
            "files_by_category": {},
            "data_distribution": {},
            "completeness_metrics": {}
        }
        
        file_inventory = self.results.get("file_inventory", {})
        if "by_category" in file_inventory:
            for category, data in file_inventory["by_category"].items():
                stats["files_by_category"][category] = {
                    "count": data["file_count"],
                    "size_gb": data["total_size_gb"],
                    "percentage": data["percentage_of_total"]
                }
        
        dicom = self.results.get("dicom_catalog", {})
        if dicom.get("hierarchy"):
            stats["data_distribution"]["dicom"] = {
                "patients": dicom["hierarchy"]["patient_count"],
                "studies": dicom["hierarchy"]["study_count"],
                "series": dicom["hierarchy"]["series_count"],
                "instances": dicom["hierarchy"]["instance_count"]
            }
        
        tabular = self.results.get("tabular_catalog", {})
        if tabular.get("file_count", 0) > 0:
            stats["data_distribution"]["tabular"] = {
                "files": tabular["file_count"],
                "total_rows": tabular.get("total_rows", 0),
                "total_columns": tabular.get("total_columns", 0)
            }
        
        return stats
    
    def _generate_tcia_form_helper(self) -> Dict[str, Any]:
        """Generate pre-filled responses for TCIA submission form."""
        self._log("\n📝 Generating TCIA form helper data...")
        
        dicom = self.results.get('dicom_catalog', {})
        tabular = self.results.get('tabular_catalog', {})
        overview = self.results.get('dataset_overview', {})
        inventory = self.results.get('file_inventory', {})
        
        form_data = {
            "contact_information": {
                "question": "Provide a scientific point of contact",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please provide: Name, Email, Phone Number"
            },
            "technical_contact": {
                "question": "Provide a technical point of contact",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please provide: Name, Email, Phone Number"
            },
            "legal_contact": {
                "question": "Provide a legal/contracts administrator",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please provide authorized signatory name and email"
            },
            "time_constraints": {
                "question": "Are there any time constraints associated with sharing your data set?",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please specify any deadlines or time constraints"
            },
            "dataset_title": {
                "question": "Suggest a descriptive title for your dataset",
                "status": "REQUIRES_MANUAL_INPUT",
                "suggested_format": f"[Disease/Condition] [Imaging Modality] Dataset with [Key Feature]",
                "answer": "Please provide a descriptive title similar to a manuscript title"
            },
            "dataset_nickname": {
                "question": "Suggest a shorter nickname for your dataset",
                "status": "REQUIRES_MANUAL_INPUT",
                "requirements": "Must be < 30 characters, letters/numbers/dashes only",
                "answer": "Please provide a short nickname (e.g., 'Cancer-CT-2024')"
            },
            "authors": {
                "question": "List the authors of this data set",
                "status": "REQUIRES_MANUAL_INPUT",
                "format": "(FAMILY, GIVEN) with ORCIDs",
                "answer": "Please list all authors with their ORCIDs from https://orcid.org"
            },
            "dataset_description": {
                "question": "Provide a Dataset Description (Abstract + Description)",
                "status": "PARTIAL_AUTO",
                "auto_generated_stats": self._generate_dataset_description(),
                "answer": "See auto_generated_stats for data summary. Please add scientific context, methodology, and usage instructions."
            },
            "previous_publication": {
                "question": "Has this data ever been published elsewhere?",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please specify if data exists elsewhere and whether it will remain accessible"
            },
            "disease_site": {
                "question": "What is the primary disease site/location of these subjects?",
                "status": "PARTIAL_AUTO",
                "detected_body_parts": self._extract_disease_sites(),
                "answer": "See detected_body_parts for automatically detected information. Please select appropriate term(s) from TCIA taxonomy."
            },
            "histologic_diagnosis": {
                "question": "What is the histologic diagnosis of these subjects?",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please choose relevant term(s) from TCIA taxonomy column B"
            },
            "image_types": {
                "question": "Which image types are included in the data set?",
                "status": "AUTO_GENERATED",
                "answer": self._list_image_types()
            },
            "supporting_data": {
                "question": "Which kinds of supporting data are included in the data set?",
                "status": "AUTO_GENERATED",
                "answer": self._list_supporting_data()
            },
            "file_formats": {
                "question": "Specify the file format utilized for each type of data",
                "status": "AUTO_GENERATED",
                "answer": self._describe_file_formats()
            },
            "number_of_subjects": {
                "question": "How many subjects are in your data set?",
                "status": "AUTO_GENERATED",
                "answer": self._extract_subject_count()
            },
            "number_of_studies": {
                "question": "How many total radiology studies or pathology slides are in your data set?",
                "status": "AUTO_GENERATED",
                "answer": self._extract_study_count()
            },
            "disk_space": {
                "question": "What is the approximate disk space that would be required to store your data set?",
                "status": "AUTO_GENERATED",
                "answer": f"{overview.get('total_size_gb', 0)} GB ({overview.get('total_size_mb', 0)} MB)"
            },
            "data_modifications": {
                "question": "Please describe any steps you've taken to modify your data prior to submitting it to us",
                "status": "PARTIAL_AUTO",
                "detected_info": self._describe_modifications(),
                "answer": "See detected_info for file format analysis. Please describe any de-identification, conversion, or registration steps performed."
            },
            "patient_faces": {
                "question": "Does your data contain any images of patient faces?",
                "status": "AUTO_GENERATED",
                "answer": self._check_for_patient_faces()
            },
            "access_policy_exceptions": {
                "question": "Do you need to request any exceptions to TCIA's Open Access & Usage Policy?",
                "status": "REQUIRES_MANUAL_INPUT",
                "default": "No exceptions requested",
                "answer": "Default: No exceptions requested. Change if controlled access is needed."
            },
            "dataset_publications": {
                "question": "Publications specifically about the dataset contents",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please provide citations for data descriptor articles"
            },
            "related_publications": {
                "question": "Additional publications with scientific findings from this data",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please provide citations for papers using this dataset"
            },
            "acknowledgments": {
                "question": "Please list any additional acknowledgments or funding statements",
                "status": "REQUIRES_MANUAL_INPUT",
                "answer": "Please provide funding sources and acknowledgments"
            },
            "publication_reason": {
                "question": "Why would you like to publish this dataset on TCIA?",
                "status": "REQUIRES_MANUAL_INPUT",
                "options": [
                    "To meet a funding agency's data sharing requirements",
                    "To meet a journal's data sharing requirements",
                    "To facilitate a collaborative project",
                    "To facilitate a challenge competition"
                ],
                "answer": "Please select all that apply from the options above"
            }
        }
        
        return form_data
    
    def _generate_dataset_description(self) -> str:
        """Generate statistical description for dataset abstract."""
        dicom = self.results.get('dicom_catalog', {})
        tabular = self.results.get('tabular_catalog', {})
        overview = self.results.get('dataset_overview', {})
        
        description_parts = []
        
        # Dataset size
        description_parts.append(f"Dataset Size: {overview.get('total_size_gb', 0)} GB across {overview.get('total_files', 0):,} files")
        
        # DICOM information
        if dicom.get('hierarchy', {}).get('patient_count', 0) > 0:
            hier = dicom['hierarchy']
            description_parts.append(
                f"\nDICOM Data: {hier['patient_count']} patients, "
                f"{hier['study_count']} studies, "
                f"{hier['series_count']} series, "
                f"{hier['instance_count']:,} instances"
            )
            
            modalities = dicom.get('modalities_found', {})
            if modalities:
                mod_str = ", ".join([f"{mod} ({count})" for mod, count in list(modalities.items())[:10]])
                description_parts.append(f"Modalities: {mod_str}")
        
        # Tabular information
        if tabular.get('file_count', 0) > 0:
            description_parts.append(
                f"\nClinical Data: {tabular['file_count']} tabular files with "
                f"{tabular.get('total_rows', 0):,} total rows"
            )
        
        return "\n".join(description_parts)
    
    def _extract_disease_sites(self) -> Dict[str, Any]:
        """Extract potential disease sites from DICOM metadata."""
        dicom = self.results.get('dicom_catalog', {})
        
        body_parts = dicom.get('body_parts_examined', {})
        study_descriptions = dicom.get('study_descriptions_found', {})
        
        # Map common body parts to disease sites
        body_part_mapping = {
            'BRAIN': ['Brain', 'Head and Neck'],
            'HEAD': ['Head and Neck'],
            'CHEST': ['Lung', 'Breast', 'Heart'],
            'LUNG': ['Lung'],
            'BREAST': ['Breast'],
            'ABDOMEN': ['Abdomen', 'Liver', 'Pancreas', 'Kidney'],
            'PELVIS': ['Pelvis', 'Prostate', 'Bladder', 'Colon'],
            'SPINE': ['Spine'],
            'EXTREMITY': ['Bone', 'Soft Tissue']
        }
        
        suggested_sites = []
        for body_part in body_parts.keys():
            bp_upper = body_part.upper()
            for key, sites in body_part_mapping.items():
                if key in bp_upper:
                    suggested_sites.extend(sites)
        
        return {
            "body_parts_found": list(body_parts.keys()),
            "suggested_disease_sites": list(set(suggested_sites)),
            "study_descriptions_sample": list(study_descriptions.keys())[:10]
        }
    
    def _list_image_types(self) -> str:
        """List all image types found in dataset."""
        dicom = self.results.get('dicom_catalog', {})
        inventory = self.results.get('file_inventory', {}).get('by_category', {})
        
        image_types = []
        
        # DICOM modalities
        if dicom.get('modalities_found'):
            for modality in dicom['modalities_found'].keys():
                if modality in ['CT', 'MR', 'PT', 'NM']:
                    image_types.append(modality)
                elif modality in ['US']:
                    image_types.append('Ultrasound')
                elif modality in ['MG', 'DM']:
                    image_types.append('Mammograms')
                elif modality in ['DX', 'CR']:
                    image_types.append('X-ray')
        
        # Radiotherapy
        if dicom.get('radiotherapy', {}).get('has_rt_data'):
            image_types.append('Radiation Therapy (RTSTRUCT, RTDOSE, RTPLAN)')
        
        # Pathology slides
        if inventory.get('Pathology_Slides', {}).get('file_count', 0) > 0:
            image_types.append('Whole Slide Image')
        
        # Standard images
        if inventory.get('Images', {}).get('file_count', 0) > 0:
            image_types.append('Standard Images (JPG/PNG/etc)')
        
        if not image_types:
            return "No imaging data detected"
        
        return "Image types found:\n" + "\n".join([f"  ✓ {img_type}" for img_type in set(image_types)])
    
    def _list_supporting_data(self) -> str:
        """List all supporting data types found."""
        tabular = self.results.get('tabular_catalog', {})
        inventory = self.results.get('file_inventory', {}).get('by_category', {})
        
        supporting_data = []
        
        # Clinical data
        if tabular.get('file_count', 0) > 0:
            supporting_data.append(
                f"Clinical data: {tabular['file_count']} files "
                f"(demographics, outcomes, treatment details)"
            )
        
        # Image analyses
        dicom = self.results.get('dicom_catalog', {})
        if dicom.get('radiotherapy', {}).get('has_rt_data'):
            supporting_data.append("Image Analyses: Segmentations (RTSTRUCT)")
        
        # Code/Scripts
        if inventory.get('Code_Scripts', {}).get('file_count', 0) > 0:
            supporting_data.append(
                f"Software/Source Code: {inventory['Code_Scripts']['file_count']} files"
            )
        
        # Documents
        if inventory.get('Documents', {}).get('file_count', 0) > 0:
            supporting_data.append(
                f"Documentation: {inventory['Documents']['file_count']} files"
            )
        
        if not supporting_data:
            return "No additional data (only images)"
        
        return "Supporting data found:\n" + "\n".join([f"  ✓ {data}" for data in supporting_data])
    
    def _describe_file_formats(self) -> str:
        """Describe file formats for each data type."""
        inventory = self.results.get('file_inventory', {}).get('by_category', {})
        
        format_descriptions = []
        
        for category, data in inventory.items():
            if data['file_count'] > 0:
                exts = ", ".join(data['extensions'][:5])  # First 5 extensions
                format_descriptions.append(
                    f"{category}: {exts} format ({data['file_count']} files, {data['total_size_gb']} GB)"
                )
        
        return "\n".join(format_descriptions)
    
    def _extract_subject_count(self) -> str:
        """Extract total number of unique subjects."""
        dicom = self.results.get('dicom_catalog', {})
        tabular = self.results.get('tabular_catalog', {})
        
        subjects = set()
        
        # From DICOM
        dicom_count = dicom.get('hierarchy', {}).get('patient_count', 0)
        if dicom_count > 0:
            subjects.add(f"DICOM: {dicom_count} patients")
        
        # From tabular data
        patient_cols = tabular.get('patient_id_columns', [])
        if patient_cols:
            for col_info in patient_cols:
                subjects.add(f"CSV ({col_info['file']}): {col_info['unique_count']} unique {col_info['column']}")
        
        if not subjects:
            return "Unable to determine - please count manually"
        
        return "\n".join(sorted(subjects))
    
    def _extract_study_count(self) -> str:
        """Extract total number of studies."""
        dicom = self.results.get('dicom_catalog', {})
        inventory = self.results.get('file_inventory', {}).get('by_category', {})
        
        study_info = []
        
        # DICOM studies
        if dicom.get('hierarchy', {}).get('study_count', 0) > 0:
            hier = dicom['hierarchy']
            study_info.append(f"DICOM Studies: {hier['study_count']}")
            study_info.append(f"  - {hier['series_count']} series total")
            study_info.append(f"  - {hier['instance_count']:,} instances total")
            
            modalities = dicom.get('modalities_found', {})
            for mod, count in list(modalities.items())[:10]:
                study_info.append(f"  - {mod}: {count} instances")
        
        # Pathology slides
        if inventory.get('Pathology_Slides', {}).get('file_count', 0) > 0:
            study_info.append(
                f"Pathology Slides: {inventory['Pathology_Slides']['file_count']} slides"
            )
        
        if not study_info:
            return "No studies detected - please count manually"
        
        return "\n".join(study_info)
    
    def _describe_modifications(self) -> str:
        """Describe detected file format information."""
        inventory = self.results.get('file_inventory', {}).get('by_category', {})
        
        observations = []
        
        # Check for DICOM
        if inventory.get('DICOM', {}).get('file_count', 0) > 0:
            observations.append("✓ Data includes DICOM files (native format)")
        
        # Check for converted formats
        if inventory.get('Medical_Imaging', {}).get('file_count', 0) > 0:
            exts = inventory['Medical_Imaging']['extensions']
            observations.append(f"✓ Converted medical imaging formats detected: {', '.join(exts)}")
        
        # Check for tabular data
        if inventory.get('Tabular_Data', {}).get('file_count', 0) > 0:
            observations.append("✓ Clinical data in tabular format")
        
        observations.append("\n⚠️ Please describe any de-identification, conversion, or registration steps you performed")
        
        return "\n".join(observations)
    
    def _check_for_patient_faces(self) -> str:
        """Determine if dataset likely contains patient face images."""
        dicom = self.results.get('dicom_catalog', {})
        
        if not dicom.get('modalities_found'):
            return "No - No DICOM data found"
        
        # Check for modalities that could contain faces
        risky_modalities = {'CT', 'MR', 'PT', 'NM'}
        found_modalities = set(dicom.get('modalities_found', {}).keys())
        
        if not (risky_modalities & found_modalities):
            return "No - Modalities detected do not typically include facial imaging"
        
        # Check body parts
        body_parts = dicom.get('body_parts_examined', {})
        risky_parts = {'HEAD', 'FACE', 'WHOLEBODY', 'SKULL', 'BRAIN', 'NECK'}
        
        detected_risky_parts = []
        for body_part in body_parts.keys():
            bp_upper = body_part.upper()
            for risky in risky_parts:
                if risky in bp_upper:
                    detected_risky_parts.append(body_part)
                    break
        
        if detected_risky_parts:
            return (f"⚠️ LIKELY YES - Dataset contains {', '.join(found_modalities & risky_modalities)} imaging "
                   f"of: {', '.join(detected_risky_parts)}\n"
                   f"These scans may allow 3D facial reconstruction. "
                   f"Please discuss de-identification options with TCIA.")
        
        return ("POSSIBLE - Dataset contains 3D imaging modalities. "
               "Please verify if scans include head/face regions.")
    
    def print_comprehensive_summary(self):
        """Print detailed, formatted summary of the entire dataset."""
        print("\n" + "="*100)
        print("COMPREHENSIVE DATASET ANALYSIS REPORT")
        print("="*100)
        
        metadata = self.results['metadata']
        overview = self.results['dataset_overview']
        structure = self.results['directory_structure']
        inventory = self.results['file_inventory']
        
        # Dataset Overview
        print("\n" + "─"*100)
        print("📦 DATASET OVERVIEW")
        print("─"*100)
        print(f"Dataset Name: {metadata['dataset_name']}")
        print(f"Dataset Path: {metadata['dataset_path']}")
        print(f"Analysis Date: {metadata['analysis_timestamp']}")
        print(f"Analysis Duration: {metadata.get('analysis_duration_seconds', 0)} seconds")
        print(f"\nTotal Files: {overview['total_files']:,}")
        print(f"Total Size: {overview['total_size_gb']:.2f} GB ({overview['total_size_mb']:.2f} MB)")
        print(f"Average File Size: {overview['average_file_size_mb']:.4f} MB")
        
        # File Size Distribution
        print(f"\nFile Size Distribution:")
        for size_cat, count in overview['files_by_size_category'].items():
            percentage = (count / overview['total_files'] * 100) if overview['total_files'] > 0 else 0
            print(f"  {size_cat:>12}: {count:>6,} files ({percentage:>5.1f}%)")
        
        # Directory Structure
        print("\n" + "─"*100)
        print("📁 DIRECTORY STRUCTURE")
        print("─"*100)
        print(f"Total Directories: {structure['total_directories']}")
        print(f"Maximum Depth: {structure['total_depth']} levels")
        print(f"\nDirectory Distribution by Level:")
        for level in sorted(structure['directories_by_level'].keys())[:10]:
            dirs = structure['directories_by_level'][level]
            print(f"  Level {level}: {len(dirs)} directories")
            if level < 3:  # Show first few levels
                for dir_path in dirs[:5]:
                    print(f"    └─ {dir_path}")
                if len(dirs) > 5:
                    print(f"    └─ ... and {len(dirs)-5} more")
        
        # File Type Inventory
        print("\n" + "─"*100)
        print("📋 FILE TYPE INVENTORY")
        print("─"*100)
        print(f"Total Categories: {inventory['total_categories']}")
        print(f"Unique Extensions: {inventory['total_unique_extensions']}")
        
        print(f"\n{'Category':<25} {'Files':>10} {'Size (GB)':>12} {'% of Total':>12} {'Extensions'}")
        print("─" * 100)
        
        for category, data in sorted(inventory['by_category'].items(), 
                                     key=lambda x: x[1]['file_count'], reverse=True):
            exts = ', '.join(data['extensions'][:5])
            if len(data['extensions']) > 5:
                exts += f" +{len(data['extensions'])-5} more"
            print(f"{category:<25} {data['file_count']:>10,} {data['total_size_gb']:>12.4f} "
                  f"{data['percentage_of_total']:>11.1f}% {exts}")
        
        # DICOM Analysis
        dicom = self.results.get('dicom_catalog', {})
        if dicom.get('total_files_found', 0) > 0:
            print("\n" + "─"*100)
            print("🏥 DICOM DATA ANALYSIS")
            print("─"*100)
            print(f"Total DICOM Files: {dicom['total_files_found']:,}")
            print(f"Successfully Processed: {dicom['successfully_processed']:,} ({dicom['processing_success_rate']:.1f}%)")
            
            if dicom.get('hierarchy'):
                hier = dicom['hierarchy']
                print(f"\nHierarchy:")
                print(f"  Patients:  {hier['patient_count']:>8,}")
                print(f"  Studies:   {hier['study_count']:>8,}")
                print(f"  Series:    {hier['series_count']:>8,}")
                print(f"  Instances: {hier['instance_count']:>8,}")
                print(f"\nAverages:")
                print(f"  Studies per Patient:    {hier['avg_studies_per_patient']:.2f}")
                print(f"  Series per Study:       {hier['avg_series_per_study']:.2f}")
                print(f"  Instances per Series:   {hier['avg_instances_per_series']:.2f}")
            
            if dicom.get('modalities_found'):
                print(f"\nModalities Found ({len(dicom['modalities_found'])}):")
                for modality, count in list(dicom['modalities_found'].items())[:15]:
                    print(f"  {modality:>10}: {count:>8,} instances")
            
            if dicom.get('manufacturers_found'):
                print(f"\nManufacturers ({len(dicom['manufacturers_found'])}):")
                for mfr, count in list(dicom['manufacturers_found'].items())[:10]:
                    print(f"  {mfr[:40]:<40}: {count:>6,} files")
            
            if dicom.get('body_parts_examined'):
                print(f"\nBody Parts Examined ({len(dicom['body_parts_examined'])}):")
                for bp, count in list(dicom['body_parts_examined'].items())[:15]:
                    print(f"  {bp:<30}: {count:>6,} instances")
            
            if dicom.get('radiotherapy', {}).get('has_rt_data'):
                print(f"\n⚡ Radiotherapy Data Detected:")
                for rt_type, count in dicom['radiotherapy']['rt_structure_counts'].items():
                    print(f"  {rt_type}: {count:,} files")
        
        # Tabular Data Analysis
        tabular = self.results.get('tabular_catalog', {})
        if tabular.get('file_count', 0) > 0:
            print("\n" + "─"*100)
            print("📊 TABULAR DATA ANALYSIS")
            print("─"*100)
            print(f"Total Files: {tabular['file_count']}")
            print(f"Total Rows: {tabular.get('total_rows', 0):,}")
            print(f"Total Columns: {tabular.get('total_columns', 0)}")
            print(f"Unique Column Names: {len(tabular.get('unique_column_names', []))}")
            
            print(f"\nFiles:")
            for fname, fdata in tabular.get('files', {}).items():
                if 'row_count' in fdata:
                    print(f"  {fname}:")
                    print(f"    Rows: {fdata['row_count']:,}")
                    print(f"    Columns: {fdata['column_count']}")
                    print(f"    Size: {fdata['file_size_mb']:.2f} MB")
                    if fdata.get('duplicate_row_count', 0) > 0:
                        print(f"    Duplicates: {fdata['duplicate_row_count']} ({fdata['duplicate_row_percentage']:.1f}%)")
            
            if tabular.get('patient_id_columns'):
                print(f"\n🔍 Potential Patient ID Columns:")
                for col_info in tabular['patient_id_columns']:
                    print(f"  {col_info['file']} -> {col_info['column']}: {col_info['unique_count']} unique values")
        
        # Other Data Types
        images = self.results.get('image_catalog', {})
        if images.get('file_count', 0) > 0:
            print("\n" + "─"*100)
            print("🖼️  IMAGE FILES")
            print("─"*100)
            print(f"Total Image Files: {images['file_count']}")
            print(f"Total Size: {images['total_size_mb']:.2f} MB")
            for fmt, data in images.get('formats', {}).items():
                print(f"  {fmt}: {data['file_count']} files ({data['total_size_mb']:.2f} MB)")
        
        archives = self.results.get('archive_files', {})
        if archives.get('file_count', 0) > 0:
            print("\n" + "─"*100)
            print("📦 ARCHIVE FILES")
            print("─"*100)
            print(f"Total Archives: {archives['file_count']}")
            print(f"Total Size: {archives['total_size_mb']:.2f} MB")
        
        # Relationships
        relationships = self.results.get('relationships', {})
        if relationships.get('patient_overlap'):
            print("\n" + "─"*100)
            print("🔗 DATA RELATIONSHIPS")
            print("─"*100)
            overlap = relationships['patient_overlap']
            print(f"DICOM Patients: {overlap['dicom_patient_count']}")
            print(f"CSV Patients: {overlap['csv_patient_count']}")
            print(f"Common Patients: {overlap['common_patients']}")
            print(f"DICOM Only: {overlap['dicom_only']}")
            print(f"CSV Only: {overlap['csv_only']}")
    
    def print_brief_summary(self):
        """Print a brief, comprehensive summary of the dataset."""
        print("\n" + "="*100)
        print("DATASET SUMMARY")
        print("="*100)
        
        overview = self.results['dataset_overview']
        structure = self.results['directory_structure']
        inventory = self.results['file_inventory']
        dicom = self.results.get('dicom_catalog', {})
        tabular = self.results.get('tabular_catalog', {})
        
        # Basic Info
        print(f"\n📁 DATASET: {self.results['metadata']['dataset_name']}")
        print(f"   Total Size: {overview['total_size_gb']:.2f} GB ({overview['total_files']:,} files)")
        print(f"   Directories: {structure['total_directories']} (Max depth: {structure['total_depth']} levels)")
        
        # File Categories Summary
        print(f"\n📊 FILE CATEGORIES ({inventory['total_categories']} types):")
        for category, data in sorted(inventory['by_category'].items(), 
                                     key=lambda x: x[1]['file_count'], reverse=True):
            exts = ', '.join(data['extensions'][:3])
            print(f"   • {category}: {data['file_count']:,} files ({data['total_size_gb']:.2f} GB) - {exts}")
        
        # DICOM Summary
        if dicom.get('hierarchy', {}).get('patient_count', 0) > 0:
            hier = dicom['hierarchy']
            print(f"\n🏥 DICOM DATA:")
            print(f"   Patients: {hier['patient_count']}")
            print(f"   Studies: {hier['study_count']}")
            print(f"   Series: {hier['series_count']}")
            print(f"   Instances: {hier['instance_count']:,}")
            
            if dicom.get('modalities_found'):
                mods = ', '.join([f"{k}({v})" for k, v in list(dicom['modalities_found'].items())[:8]])
                print(f"   Modalities: {mods}")
            
            if dicom.get('body_parts_examined'):
                parts = ', '.join(list(dicom['body_parts_examined'].keys())[:8])
                print(f"   Body Parts: {parts}")
        
        # Tabular Summary
        if tabular.get('file_count', 0) > 0:
            print(f"\n📋 CLINICAL DATA:")
            print(f"   Files: {tabular['file_count']}")
            print(f"   Total Rows: {tabular['total_rows']:,}")
            print(f"   Total Columns: {tabular['total_columns']}")
            if tabular.get('patient_id_columns'):
                print(f"   Patient ID Columns: {len(tabular['patient_id_columns'])} detected")
        
        # Directory Structure
        print(f"\n📂 DIRECTORY STRUCTURE:")
        for level in sorted(structure['directories_by_level'].keys())[:3]:
            dirs = structure['directories_by_level'][level]
            print(f"   Level {level}: {len(dirs)} directories")
            for d in dirs[:3]:
                print(f"      └─ {d}")
            if len(dirs) > 3:
                print(f"      └─ ... and {len(dirs)-3} more")
        
        print("\n" + "="*100)
    
    def print_tcia_form_answers(self):
        """Print the 6 definitively answerable TCIA form questions."""
        print("\n" + "="*100)
        print("TCIA SUBMISSION FORM - AUTO-GENERATED ANSWERS")
        print("="*100)
        print("\n✅ The following 6 questions CAN be automatically answered with HIGH CONFIDENCE:")
        
        form_data = self.tcia_form_data
        
        # Question 1: Image Types
        print("\n" + "─"*100)
        print("Q1: Which image types are included in the data set?")
        print("─"*100)
        print(form_data['image_types']['answer'])
        
        # Question 2: Supporting Data
        print("\n" + "─"*100)
        print("Q2: Which kinds of supporting data are included in the data set?")
        print("─"*100)
        print(form_data['supporting_data']['answer'])
        
        # Question 3: File Formats
        print("\n" + "─"*100)
        print("Q3: Specify the file format utilized for each type of data")
        print("─"*100)
        print(form_data['file_formats']['answer'])
        
        # Question 4: Number of Subjects
        print("\n" + "─"*100)
        print("Q4: How many subjects are in your data set?")
        print("─"*100)
        print(form_data['number_of_subjects']['answer'])
        
        # Question 5: Number of Studies
        print("\n" + "─"*100)
        print("Q5: How many total radiology studies or pathology slides?")
        print("─"*100)
        print(form_data['number_of_studies']['answer'])
        
        # Question 6: Disk Space
        print("\n" + "─"*100)
        print("Q6: What is the approximate disk space required?")
        print("─"*100)
        print(form_data['disk_space']['answer'])
        
        print("\n" + "="*100)
        print("NOTE: All other questions require manual human input")
        print("="*100)
    
    def save_results(self, output_path: str = None, format: str = 'json'):
        """
        Save analysis results to file.
        
        Args:
            output_path: Output file path
            format: Output format ('json' or 'html')
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"tcia_dataset_analysis_{timestamp}.{format}"
        
        try:
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            elif format == 'html':
                self._save_html_report(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self._log(f"Results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self._log(f"Error saving results: {str(e)}")
            raise
    
    def _save_html_report(self, output_path: str):
        """Generate an HTML report."""
        dicom = self.results.get("dicom_catalog", {})
        tabular = self.results.get("tabular_catalog", {})
        overview = self.results.get("dataset_overview", {})
        inventory = self.results.get("file_inventory", {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TCIA Dataset Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-top: 30px; }}
                h3 {{ color: #555; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 5px; border-left: 4px solid #3498db; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .badge {{ display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 0.85em; margin: 2px; background: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 TCIA Dataset Analysis Report</h1>
                <p><strong>Generated:</strong> {self.results['metadata']['analysis_timestamp']}</p>
                <p><strong>Dataset:</strong> {self.results['metadata']['dataset_path']}</p>
                
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{overview.get('total_files', 0):,}</div>
                            <div class="metric-label">Total Files</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{overview.get('total_size_gb', 0):.2f} GB</div>
                            <div class="metric-label">Total Size</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self.results['directory_structure'].get('total_directories', 0)}</div>
                            <div class="metric-label">Directories</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{inventory.get('total_categories', 0)}</div>
                            <div class="metric-label">File Categories</div>
                        </div>
                    </div>
                </div>
                
                <h2>File Categories</h2>
                <table>
                    <tr><th>Category</th><th>File Count</th><th>Total Size</th><th>Extensions</th></tr>
                    {"".join([f'<tr><td><strong>{cat}</strong></td><td>{data["file_count"]:,}</td><td>{data["total_size_mb"]} MB</td><td>{"".join([f"<span class=\'badge\'>{ext}</span>" for ext in data.get("extensions", [])])}</td></tr>' for cat, data in inventory.get('by_category', {}).items()])}
                </table>
                
                {f'''
                <h2>DICOM Data Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{dicom.get('total_files_found', 0):,}</div>
                        <div class="metric-label">DICOM Files</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dicom.get('hierarchy', {}).get('patient_count', 0)}</div>
                        <div class="metric-label">Patients</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dicom.get('hierarchy', {}).get('study_count', 0)}</div>
                        <div class="metric-label">Studies</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dicom.get('hierarchy', {}).get('series_count', 0)}</div>
                        <div class="metric-label">Series</div>
                    </div>
                </div>
                <h3>Modalities</h3>
                <p>{"".join([f"<span class=\'badge\'>{mod}: {count}</span>" for mod, count in list(dicom.get('modalities_found', {}).items())[:20]])}</p>
                ''' if dicom.get('total_files_found', 0) > 0 else ''}
                
                {f'''
                <h2>Tabular Data Summary</h2>
                <p><strong>Files:</strong> {tabular.get('file_count', 0)}</p>
                <p><strong>Total Rows:</strong> {tabular.get('total_rows', 0):,}</p>
                <p><strong>Total Columns:</strong> {tabular.get('total_columns', 0)}</p>
                ''' if tabular.get('file_count', 0) > 0 else ''}
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced TCIA Dataset Analyzer with Form Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a dataset
  python tcia_analyzer.py /path/to/dataset
  
  # Save JSON and HTML reports
  python tcia_analyzer.py /path/to/dataset --output analysis.json --html
  
  # Quiet mode
  python tcia_analyzer.py /path/to/dataset --quiet
        """
    )
    
    parser.add_argument(
        'dataset_path',
        nargs='?',
        help='Path to the medical dataset directory'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Also generate an HTML report'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    parser.add_argument(
        '--tcia-form-only',
        action='store_true',
        help='Only print TCIA form answers'
    )
    
    args = parser.parse_args()
    
    # Get dataset path
    dataset_path = args.dataset_path
    if not dataset_path:
        print("TCIA Dataset Analyzer")
        print("=" * 50)
        dataset_path = input("Enter the path to your medical dataset folder: ").strip()
        
        if not dataset_path:
            print("Error: No path provided")
            sys.exit(1)
    
    # Remove quotes
    dataset_path = dataset_path.strip('"').strip("'")
    
    # Validate path
    if not os.path.exists(dataset_path):
        print(f"Error: Directory does not exist: {dataset_path}")
        sys.exit(1)
    
    # Initialize analyzer
    print(f"\nInitializing TCIA Dataset Analyzer for: {dataset_path}")
    analyzer = TCIADatasetAnalyzer(
        max_workers=args.workers,
        verbose=not args.quiet
    )
    
    try:
        # Run analysis
        results = analyzer.analyze_dataset(dataset_path)
        
        # Print summaries
        if args.tcia_form_only:
            analyzer.print_tcia_form_answers()
        else:
            analyzer.print_comprehensive_summary()
            print("\n")
            analyzer.print_tcia_form_answers()
        
        # Save JSON results
        json_path = analyzer.save_results(args.output, format='json')
        print(f"\n✅ JSON analysis saved: {json_path}")
        
        # Save HTML report if requested
        if args.html:
            html_path = json_path.replace('.json', '.html')
            analyzer.save_results(html_path, format='html')
            print(f"✅ HTML report saved: {html_path}")
        
        print("\n✨ Analysis complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())