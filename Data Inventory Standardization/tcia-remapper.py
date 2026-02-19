import streamlit as st
import pandas as pd
import json
import re
import os
import sys
import requests
import base64
from io import BytesIO
import importlib.util

# Add tcia-remapping-skill to the path and import the helper
skill_dir = os.path.join(os.path.dirname(__file__), 'tcia-remapping-skill')
remap_helper_path = os.path.join(skill_dir, 'remap_helper.py')
spec = importlib.util.spec_from_file_location("remap_helper", remap_helper_path)
remap_helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remap_helper)

# Import MDF parser
mdf_parser_path = os.path.join(skill_dir, 'mdf_parser.py')
spec_mdf = importlib.util.spec_from_file_location("mdf_parser", mdf_parser_path)
mdf_parser = importlib.util.module_from_spec(spec_mdf)
spec_mdf.loader.exec_module(mdf_parser)

# Import functions
load_json = remap_helper.load_json
get_closest_match = remap_helper.get_closest_match
validate_dataframe = remap_helper.validate_dataframe
split_data_by_schema = remap_helper.split_data_by_schema
write_metadata_tsv = remap_helper.write_metadata_tsv
check_metadata_conflict = remap_helper.check_metadata_conflict
get_mdf_resources = mdf_parser.get_mdf_resources

st.set_page_config(page_title="TCIA Dataset Remapper", layout="wide")

# Load resources
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), 'tcia-remapping-skill', 'resources')
SCHEMA_FILE = os.path.join(RESOURCES_DIR, 'schema.json')
PERMISSIBLE_VALUES_FILE = os.path.join(RESOURCES_DIR, 'permissible_values.json')

# Default programs with complete metadata
DEFAULT_PROGRAMS = {
    "Community": {
        "program_name": "Community",
        "program_short_name": "Community",
        "institution_name": "",
        "program_short_description": "Community-contributed imaging collections",
        "program_full_description": "The Community program encompasses imaging collections contributed by individual researchers and institutions that are not part of larger organized programs.",
        "program_external_url": "https://www.cancerimagingarchive.net/"
    },
    "TCGA": {
        "program_name": "The Cancer Genome Atlas",
        "program_short_name": "TCGA",
        "institution_name": "National Cancer Institute",
        "program_short_description": "A landmark cancer genomics program",
        "program_full_description": "The Cancer Genome Atlas (TCGA) is a landmark cancer genomics program that molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types.",
        "program_external_url": "https://www.cancer.gov/tcga"
    },
    "CPTAC": {
        "program_name": "Clinical Proteomic Tumor Analysis Consortium",
        "program_short_name": "CPTAC",
        "institution_name": "National Cancer Institute",
        "program_short_description": "A comprehensive and coordinated effort to accelerate proteogenomic cancer research",
        "program_full_description": "The Clinical Proteomic Tumor Analysis Consortium (CPTAC) is a comprehensive and coordinated effort to accelerate the understanding of the molecular basis of cancer through the application of large-scale proteome and genome analysis (proteogenomics).",
        "program_external_url": "https://proteomics.cancer.gov/programs/cptac"
    },
    "APOLLO": {
        "program_name": "Applied Proteogenomics OrganizationaL Learning and Outcomes",
        "program_short_name": "APOLLO",
        "institution_name": "National Cancer Institute",
        "program_short_description": "Network for proteogenomic characterization of cancer",
        "program_full_description": "The Applied Proteogenomics OrganizationaL Learning and Outcomes (APOLLO) Network aims to generate proteogenomic data and develop analytical tools to advance precision oncology.",
        "program_external_url": "https://proteomics.cancer.gov/programs/apollo-network"
    },
    "Biobank": {
        "program_name": "Cancer Imaging Biobank",
        "program_short_name": "Biobank",
        "institution_name": "",
        "program_short_description": "Organized collections of cancer imaging data",
        "program_full_description": "The Cancer Imaging Biobank program organizes and maintains curated collections of cancer imaging data for research purposes.",
        "program_external_url": "https://www.cancerimagingarchive.net/"
    }
}

def lookup_doi(doi):
    """Fetch metadata from Crossref API"""
    if not doi:
        return None
    try:
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()['message']

            # Extract title
            title = data.get('title', [''])[0]

            # Extract authors
            authors_list = data.get('author', [])
            authors = ", ".join([f"{a.get('family', '')} {a.get('given', '')}".strip() for a in authors_list])
            if len(authors_list) > 3:
                authors = f"{authors_list[0].get('family', '')} et al."

            # Extract year
            year = ""
            issued = data.get('issued', {}).get('date-parts', [[None]])[0][0]
            if issued:
                year = str(issued)

            # Extract journal
            journal = data.get('container-title', [''])[0]

            return {
                'title': title,
                'authors': authors,
                'year': year,
                'journal': journal
            }
    except Exception as e:
        st.error(f"Error looking up DOI: {e}")
    return None

def lookup_orcid(orcid_id):
    """Fetch metadata from ORCID API"""
    if not orcid_id:
        return None
    try:
        headers = {"Accept": "application/json"}
        # Get personal details
        url_p = f"https://pub.orcid.org/v3.0/{orcid_id}/personal-details"
        res_p = requests.get(url_p, headers=headers, timeout=10)

        # Get employments
        url_e = f"https://pub.orcid.org/v3.0/{orcid_id}/employments"
        res_e = requests.get(url_e, headers=headers, timeout=10)

        metadata = {}

        if res_p.status_code == 200:
            data_p = res_p.json()
            name = data_p.get('name', {})
            metadata['first_name'] = name.get('given-names', {}).get('value', '')
            metadata['last_name'] = name.get('family-name', {}).get('value', '')

        if res_e.status_code == 200:
            data_e = res_e.json()
            affiliations = data_e.get('affiliation-group', [])
            if affiliations:
                # Get the first (likely most recent) employment
                org = affiliations[0].get('summaries', [{}])[0].get('employment-summary', {}).get('organization', {})
                metadata['organization'] = org.get('name', '')

        return metadata if metadata else None
    except Exception as e:
        st.error(f"Error looking up ORCID: {e}")
    return None

@st.cache_data
def load_resources():
    # Try loading from MDF first
    schema, mdf_pv = get_mdf_resources(RESOURCES_DIR)
    
    # Load legacy permissible values
    legacy_pv = load_json(PERMISSIBLE_VALUES_FILE)
    
    if schema:
        # Merge permissible values: MDF Enums take precedence, but legacy covers missing ones
        final_pv = legacy_pv.copy()
        if mdf_pv:
            for k, v in mdf_pv.items():
                final_pv[k] = v
        return schema, final_pv
        
    # Fallback to legacy JSON files
    st.warning("⚠️ Using legacy schema files. MDF model files not found or invalid.")
    schema = load_json(SCHEMA_FILE)
    return schema, legacy_pv

def render_dynamic_form(entity_name, schema, permissible_values, current_data=None, excluded_fields=None, custom_labels=None, disabled=False, priority_fields=None):
    """
    Renders a dynamic form for an entity based on the schema.
    """
    if excluded_fields is None:
        excluded_fields = []
    if custom_labels is None:
        custom_labels = {}
    if current_data is None:
        current_data = {}
    if priority_fields is None:
        priority_fields = []

    entity_props = schema.get(entity_name, [])
    form_data = {}

    # Filter out excluded fields and ID fields
    props_to_show = [
        p for p in entity_props 
        if p['Property'] not in excluded_fields 
        and not p['Property'].endswith('_id')
        and '.' not in p['Property']
    ]

    # Reorder based on priority
    def get_priority(p):
        name = p['Property']
        if name in priority_fields:
            return priority_fields.index(name)
        return len(priority_fields) + 1

    props_to_show.sort(key=get_priority)

    for prop in props_to_show:
        prop_name = prop['Property']
        label = custom_labels.get(prop_name, prop_name.replace('_', ' ').title())
        is_required = prop.get('Required/optional') == 'R'
        if is_required:
            label += "*"
        
        help_text = prop.get('Description', '')
        default_val = current_data.get(prop_name, "")

        if prop_name in permissible_values:
            options = permissible_values[prop_name]
            # Handle list of dicts from MDF parser
            if options and isinstance(options[0], dict):
                option_labels = [f"{o['value']}" for o in options]
                if not is_required:
                    option_labels = [""] + option_labels
                
                # Find index of default value
                current_val = str(default_val) if default_val else ""
                try:
                    default_idx = option_labels.index(current_val)
                except ValueError:
                    default_idx = 0
                
                selected = st.selectbox(label, options=option_labels, index=default_idx, help=help_text, disabled=disabled)
                form_data[prop_name] = selected
            else:
                if not is_required:
                    options = [""] + options
                try:
                    default_idx = options.index(default_val)
                except ValueError:
                    default_idx = 0
                selected = st.selectbox(label, options=options, index=default_idx, help=help_text, disabled=disabled)
                form_data[prop_name] = selected
        elif "description" in prop_name or "abstract" in prop_name or "acknowledgements" in prop_name:
            form_data[prop_name] = st.text_area(label, value=str(default_val), help=help_text, disabled=disabled)
        elif "number" in prop_name or "count" in prop_name or "size" in prop_name:
            try:
                dv = int(default_val) if default_val else 0
            except:
                dv = 0
            form_data[prop_name] = st.number_input(label, value=dv, help=help_text, disabled=disabled)
        else:
            form_data[prop_name] = st.text_input(label, value=str(default_val), help=help_text, disabled=disabled)

    return form_data

def reset_app():
    """Reset all session state"""
    keys_to_keep = []
    keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
    for key in keys_to_remove:
        del st.session_state[key]
    st.session_state.phase = 0
    st.session_state.phase0_step = 'Start'

# Initialize session state
if 'phase' not in st.session_state:
    st.session_state.phase = 0  # 0: Dataset-level metadata, 1: Structure mapping, 2: Value standardization
    st.session_state.metadata = {
        'Program': [],
        'Dataset': [],
        'Investigator': [],
        'Related_Work': []
    }
    st.session_state.phase0_step = 'Start'
    st.session_state.uploaded_data = None
    st.session_state.column_mapping = {}
    st.session_state.structure_approved = False
    st.session_state.output_dir = 'output'
    st.session_state.cicadas = {
        'abstract': '',
        'introduction': '',
        'methods_subjects': '',
        'methods_acquisition': '',
        'methods_analysis': '',
        'usage_notes': '',
        'external_resources': ''
    }
    st.session_state.generated_tsv_files = []
    # UI helper keys
    st.session_state.rw_doi = ""
    st.session_state.rw_title = ""
    st.session_state.rw_authors = ""
    st.session_state.inv_orcid = ""
    st.session_state.inv_first = ""
    st.session_state.inv_last = ""
    st.session_state.inv_org = ""

# Create output directory
if not os.path.exists(st.session_state.output_dir):
    os.makedirs(st.session_state.output_dir)

# Load schema and permissible values
schema, permissible_values = load_resources()

# Title and intro
st.title("🗂️ TCIA Dataset Remapper")
st.markdown("""
Welcome to the TCIA Dataset Remapper! This tool helps you transform your clinical and imaging research data 
into the standardized TCIA data model.
""")

# Show current phase
phase_names = ["Phase 0: Dataset-Level Metadata", "Phase 1: Structure Mapping", "Phase 2: Value Standardization"]
st.sidebar.title("Progress")
st.sidebar.write(f"**Current Phase:** {phase_names[st.session_state.phase]}")

if st.sidebar.button("🔄 Reset App"):
    reset_app()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
if st.sidebar.button("📋 Phase 0: Metadata"):
    st.session_state.phase = 0
    st.rerun()
if st.sidebar.button("🔗 Phase 1: Structure"):
    st.session_state.phase = 1
    st.rerun()
if st.sidebar.button("✅ Phase 2: Values"):
    st.session_state.phase = 2
    st.rerun()

# ============================================================================
# PHASE 0: DATASET-LEVEL METADATA COLLECTION
# ============================================================================
if st.session_state.phase == 0:
    st.header("Phase 0: Dataset-Level Metadata Collection")
    st.markdown("""
    Before remapping your source files, let's collect high-level metadata for your submission.
    We'll go through this one entity at a time: **Start → Program → Dataset → Investigator → Related Work**
    """)
    
    phase0_options = {
        "Start": "🚀 Start",
        "Program": "📁 Program",
        "Dataset": "📊 Dataset",
        "CICADAS": "📋 CICADAS",
        "Investigator": "👤 Investigator",
        "Related_Work": "📚 Related Work",
        "Review": "📝 Review & Generate"
    }
    
    # Initialize phase0_step if not in right format
    if st.session_state.phase0_step not in phase0_options:
        st.session_state.phase0_step = "Program"

    # Use a radio button to simulate tabs for programmatic control
    current_step = st.radio(
        "Navigation",
        options=list(phase0_options.keys()),
        format_func=lambda x: phase0_options[x],
        index=list(phase0_options.keys()).index(st.session_state.phase0_step),
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Update state if changed via radio
    if current_step != st.session_state.phase0_step:
        st.session_state.phase0_step = current_step
        st.rerun()

    st.markdown("---")

    # TAB 0: Start
    if st.session_state.phase0_step == "Start":
        st.subheader("Welcome to Phase 0")
        st.markdown("""
        Before we begin, would you like to import your TCIA Dataset Proposal Form?
        Importing a proposal will automatically fill in many of the fields for you, saving you time.
        """)

        import_file = st.file_uploader("📥 Import TCIA Proposal Form (TSV)", type=['tsv'])

        if import_file:
            try:
                import_df = pd.read_csv(import_file, sep='\t')
                if not import_df.empty:
                    proposal_data = import_df.iloc[0].to_dict()

                    # Map Dataset
                    ds_data = {
                        'dataset_long_name': proposal_data.get('Title', ''),
                        'dataset_short_name': proposal_data.get('Nickname', ''),
                        'dataset_abstract': proposal_data.get('Abstract', ''),
                        'number_of_participants': proposal_data.get('num_subjects', 0)
                    }
                    st.session_state.metadata['Dataset'] = [ds_data]

                    # Update CICADAS abstract
                    st.session_state.cicadas['abstract'] = proposal_data.get('Abstract', '')

                    # Map Program (Default to Community)
                    st.session_state.metadata['Program'] = [DEFAULT_PROGRAMS['Community']]

                    # Map Investigators (Authors)
                    authors_raw = str(proposal_data.get('Authors', ''))
                    new_investigators = []
                    author_entries = re.split(r'[;\n]', authors_raw)
                    for entry in author_entries:
                        entry = entry.strip()
                        if not entry: continue

                        orcid_match = re.search(r'(\d{4}-\d{4}-\d{4}-\d{3}[\dX])', entry)
                        orcid = orcid_match.group(1) if orcid_match else ""
                        name_part = re.sub(r'\(?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\)?', '', entry).strip()
                        if name_part.startswith('(') and name_part.endswith(')'):
                            name_part = name_part[1:-1].strip()

                        parts = name_part.split(',')
                        last_name = parts[0].strip() if len(parts) > 0 else ""
                        first_name = parts[1].strip() if len(parts) > 1 else ""

                        if first_name or last_name:
                            new_investigators.append({
                                'first_name': first_name,
                                'last_name': last_name,
                                'person_orcid': orcid,
                                'email': '',
                                'organization_name': ''
                            })
                    if new_investigators:
                        st.session_state.metadata['Investigator'] = new_investigators

                    # Map Related Work
                    rel_works = []
                    for k in ['citation_primary', 'citations_content', 'additional_publications']:
                        val = proposal_data.get(k)
                        if val and str(val).strip():
                            rel_works.append({
                                'publication_title': str(val).strip(),
                                'publication_type': 'Journal Article',
                                'authorship': '',
                                'DOI': ''
                            })
                    if rel_works:
                        st.session_state.metadata['Related_Work'] = rel_works

                    st.success("✅ Proposal imported successfully!")
                    st.info("The metadata has been pre-populated. Click 'Proceed' below to verify the information in each section.")
            except Exception as e:
                st.error(f"Import failed: {e}")

        st.markdown("---")
        if st.button("Proceed to Metadata Collection →", use_container_width=True, type="primary"):
            st.session_state.phase0_step = 'Program'
            st.rerun()

    # TAB 1: Program
    elif st.session_state.phase0_step == "Program":
        st.subheader("Program Information")
        st.info("""
        **Steering:** Most users should use "Community" as their program unless they are part of a major 
        NCI/NIH program (e.g., TCGA, CPTAC, APOLLO, Biobank).
        """)
        
        # Program selection
        program_options = ["(Select a Program)"] + list(DEFAULT_PROGRAMS.keys()) + ["➕ Create New Program"]

        # Determine current index
        current_idx = 0
        if st.session_state.metadata['Program']:
            prog_name = st.session_state.metadata['Program'][0].get('program_short_name')
            if prog_name in DEFAULT_PROGRAMS:
                current_idx = list(DEFAULT_PROGRAMS.keys()).index(prog_name) + 1
            else:
                current_idx = len(program_options) - 1

        program_choice = st.selectbox(
            "Select Program",
            options=program_options,
            index=current_idx,
            help="Choose a pre-defined program or create a custom one."
        )

        if program_choice != "(Select a Program)":
            is_custom = program_choice == "➕ Create New Program"
            
            if not is_custom:
                prog_data = DEFAULT_PROGRAMS[program_choice]
            else:
                prog_data = st.session_state.metadata['Program'][0] if st.session_state.metadata['Program'] else {}

            with st.form("program_form"):
                # Always render dynamic form but if not custom, the fields are disabled
                new_program_data = render_dynamic_form(
                    "Program", 
                    schema, 
                    permissible_values, 
                    current_data=prog_data,
                    disabled=not is_custom
                )

                submitted = st.form_submit_button("Save & Next")
                if submitted:
                    st.session_state.metadata['Program'] = [new_program_data]
                    st.toast("✅ Program information saved!")
                    st.session_state.phase0_step = 'Dataset'
                    st.rerun()
    
    # TAB 2: Dataset
    elif st.session_state.phase0_step == "Dataset":
        st.subheader("Dataset Information")
        
        current_ds_data = st.session_state.metadata['Dataset'][0] if st.session_state.metadata['Dataset'] else {}
        
        with st.form("dataset_form"):
            dataset_data = render_dynamic_form(
                "Dataset",
                schema,
                permissible_values,
                current_data=current_ds_data,
                excluded_fields=['dataset_description', 'dataset_abstract'],
                priority_fields=['dataset_long_name', 'dataset_short_name', 'number_of_participants']
            )
            
            submitted = st.form_submit_button("Save & Next")
            if submitted:
                # Keep existing description and abstract if they exist
                dataset_data['dataset_description'] = current_ds_data.get('dataset_description', '')
                dataset_data['dataset_abstract'] = current_ds_data.get('dataset_abstract', '')

                st.session_state.metadata['Dataset'] = [dataset_data]
                st.toast("✅ Basic Dataset information saved!")
                st.session_state.phase0_step = 'CICADAS'
                st.rerun()

    # TAB 3: CICADAS
    elif st.session_state.phase0_step == "CICADAS":
        st.subheader("CICADAS Dataset Description")
        st.markdown("""
        Follow the [CICADAS checklist](https://cancerimagingarchive.net/cicadas) to ensure your dataset
        is comprehensive and optimally discoverable.
        """)

        with st.form("cicadas_form"):
            st.write("### Abstract")
            c_abstract = st.text_area(
                "Abstract (Max 1,000 Characters)*",
                value=st.session_state.cicadas.get('abstract', ''),
                help="Brief overview of the dataset: subjects, imaging types, potential applications.",
                max_chars=1000
            )

            st.write("### Introduction")
            c_intro = st.text_area(
                "Introduction",
                value=st.session_state.cicadas.get('introduction', ''),
                help="Purpose and uniqueness of the dataset."
            )

            st.write("### Methods")
            c_m_subjects = st.text_area(
                "Subject Inclusion and Exclusion Criteria",
                value=st.session_state.cicadas.get('methods_subjects', ''),
                help="Demographics, clinical characteristics, and potential study bias."
            )
            c_m_acquisition = st.text_area(
                "Data Acquisition",
                value=st.session_state.cicadas.get('methods_acquisition', ''),
                help="Scanner details, sequence parameters, radiotracers, etc."
            )
            c_m_analysis = st.text_area(
                "Data Analysis",
                value=st.session_state.cicadas.get('methods_analysis', ''),
                help="Conversions, preprocessing, annotation protocols, quality control."
            )

            st.write("### Usage Notes")
            c_usage = st.text_area(
                "Usage Notes",
                value=st.session_state.cicadas.get('usage_notes', ''),
                help="Data organization, naming conventions, recommended software."
            )

            st.write("### External Resources")
            c_ext = st.text_area(
                "External Resources (Optional)",
                value=st.session_state.cicadas.get('external_resources', ''),
                help="Links to code, related datasets, or other tools."
            )

            submitted = st.form_submit_button("Save & Next")
            if submitted:
                # Update CICADAS state
                st.session_state.cicadas = {
                    'abstract': c_abstract,
                    'introduction': c_intro,
                    'methods_subjects': c_m_subjects,
                    'methods_acquisition': c_m_acquisition,
                    'methods_analysis': c_m_analysis,
                    'usage_notes': c_usage,
                    'external_resources': c_ext
                }

                # Construct dataset_description
                desc_parts = []
                if c_intro:
                    desc_parts.append(f"## Introduction\n{c_intro}")

                methods_content = ""
                if c_m_subjects:
                    methods_content += f"### Subject Inclusion and Exclusion Criteria\n{c_m_subjects}\n\n"
                if c_m_acquisition:
                    methods_content += f"### Data Acquisition\n{c_m_acquisition}\n\n"
                if c_m_analysis:
                    methods_content += f"### Data Analysis\n{c_m_analysis}\n\n"

                if methods_content:
                    desc_parts.append(f"## Methods\n{methods_content}")

                if c_usage:
                    desc_parts.append(f"## Usage Notes\n{c_usage}")

                if c_ext:
                    desc_parts.append(f"## External Resources\n{c_ext}")

                full_description = "\n\n".join(desc_parts)

                # Update metadata if Dataset exists
                if st.session_state.metadata['Dataset']:
                    st.session_state.metadata['Dataset'][0]['dataset_abstract'] = c_abstract
                    st.session_state.metadata['Dataset'][0]['dataset_description'] = full_description
                else:
                    st.warning("⚠️ Please fill out the basic Dataset information first.")

                st.toast("✅ CICADAS information saved!")
                st.session_state.phase0_step = 'Investigator'
                st.rerun()

    # TAB 4: Investigator
    elif st.session_state.phase0_step == "Investigator":
        st.subheader("Investigator Information")
        st.markdown("Add one or more investigators for this dataset.")
        
        # Display existing investigators
        if st.session_state.metadata['Investigator']:
            st.write("**Current Investigators:**")
            for idx, inv in enumerate(st.session_state.metadata['Investigator']):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.write(f"{idx+1}. {inv.get('first_name', '')} {inv.get('last_name', '')} ({inv.get('email', '')}) - {inv.get('organization_name', '')}")
                with col2:
                    if st.button("🗑️", key=f"del_inv_{idx}"):
                        st.session_state.metadata['Investigator'].pop(idx)
                        st.rerun()
        
        st.markdown("---")
        st.write("**Add New Investigator:**")
        
        col_orcid, col_lookup_orc = st.columns([3, 1])
        with col_orcid:
            orcid_input = st.text_input("ORCID (Optional)", value=st.session_state.get('inv_orcid', ''), help="e.g., 0000-0002-1825-0097")
        with col_lookup_orc:
            st.write(" ") # alignment
            st.write(" ")
            if st.button("🔍 Lookup ORCID"):
                orcid_metadata = lookup_orcid(orcid_input)
                if orcid_metadata:
                    st.session_state.inv_orcid = orcid_input
                    st.session_state.inv_first = orcid_metadata.get('first_name', '')
                    st.session_state.inv_last = orcid_metadata.get('last_name', '')
                    st.session_state.inv_org = orcid_metadata.get('organization', '')
                    st.success("Metadata found!")
                    st.rerun()
                else:
                    st.error("ORCID not found or no public profile.")

        with st.form("investigator_form"):
            inv_prepopulate = {
                'first_name': st.session_state.get('inv_first', ''),
                'last_name': st.session_state.get('inv_last', ''),
                'organization_name': st.session_state.get('inv_org', ''),
                'person_orcid': orcid_input
            }
            
            investigator_data = render_dynamic_form(
                "Investigator",
                schema,
                permissible_values,
                current_data=inv_prepopulate
            )
            
            submitted = st.form_submit_button("Save & Next")
            if submitted:
                if investigator_data.get('first_name') and investigator_data.get('last_name') and investigator_data.get('email'):
                    st.session_state.metadata['Investigator'].append(investigator_data)

                    # Clear session state
                    for key in ['inv_orcid', 'inv_first', 'inv_last', 'inv_org']:
                        if key in st.session_state:
                            st.session_state[key] = ""

                    st.toast(f"✅ Added investigator: {investigator_data['first_name']} {investigator_data['last_name']}")
                    st.session_state.phase0_step = 'Related_Work'
                    st.rerun()
                else:
                    st.error("Please fill in all required fields (First Name, Last Name, Email).")
    
    # TAB 5: Related Work
    elif st.session_state.phase0_step == "Related_Work":
        st.subheader("Related Work / Publications")
        st.markdown("Add publications, DOIs, or related work for this dataset.")
        
        # Display existing related works
        if st.session_state.metadata['Related_Work']:
            st.write("**Current Related Works:**")
            for idx, work in enumerate(st.session_state.metadata['Related_Work']):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.write(f"{idx+1}. {work.get('publication_title', '')} - DOI: {work.get('DOI', '')}")
                with col2:
                    if st.button("🗑️", key=f"del_work_{idx}"):
                        st.session_state.metadata['Related_Work'].pop(idx)
                        st.rerun()
        
        st.markdown("---")
        st.write("**Add New Related Work:**")
        
        col_doi, col_lookup = st.columns([3, 1])
        with col_doi:
            doi_input = st.text_input("DOI (Required)*", value=st.session_state.get('rw_doi', ''))
        with col_lookup:
            st.write(" ") # alignment
            st.write(" ")
            if st.button("🔍 Lookup DOI"):
                doi_metadata = lookup_doi(doi_input)
                if doi_metadata:
                    st.session_state.rw_doi = doi_input
                    st.session_state.rw_title = doi_metadata['title']
                    st.session_state.rw_authors = doi_metadata['authors']
                    # We could also use year and journal if we had fields for them
                    st.success("Metadata found!")
                    st.rerun()
                else:
                    st.error("DOI not found.")

        with st.form("related_work_form"):
            rw_prepopulate = {
                'DOI': st.session_state.get('rw_doi', doi_input),
                'publication_title': st.session_state.get('rw_title', ''),
                'authorship': st.session_state.get('rw_authors', '')
            }
            
            work_data = render_dynamic_form(
                "Related_Work",
                schema,
                permissible_values,
                current_data=rw_prepopulate
            )
            
            submitted = st.form_submit_button("Save & Next")
            if submitted:
                # Relationship Type is optional, but DOI, Title, Authorship, and Publication Type are required in schema
                if work_data.get('DOI') and work_data.get('publication_title') and work_data.get('authorship') and work_data.get('publication_type'):
                    st.session_state.metadata['Related_Work'].append(work_data)

                    # Clear session state for next entry
                    for key in ['rw_doi', 'rw_title', 'rw_authors']:
                        if key in st.session_state:
                            st.session_state[key] = ""

                    st.toast(f"✅ Added related work: {work_data['DOI']}")
                    st.session_state.phase0_step = 'Review'
                    st.rerun()
                else:
                    st.error("Please fill in all required fields (DOI, Title, Authorship, Publication Type).")
    
    # TAB 6: Review & Generate
    elif st.session_state.phase0_step == "Review":
        st.subheader("Review & Generate TSV Files")
        st.markdown("Review all your metadata and generate the TSV files.")
        
        # --- Automatic Generation ---
        generated_files_map = {}
        for entity_name, data in st.session_state.metadata.items():
            if data:
                filepath = write_metadata_tsv(entity_name, data, schema, st.session_state.output_dir)
                if filepath:
                    generated_files_map[entity_name] = filepath
        st.session_state.generated_tsv_files = list(generated_files_map.values())
        # ----------------------------

        # Display recap
        st.write("### 📋 Metadata Summary")
        
        review_entities = [
            ("Program", "Program"),
            ("Dataset", "Dataset"),
            ("Investigator", "Investigators"),
            ("Related_Work", "Related Works")
        ]
        
        for entity_key, label in review_entities:
            with st.expander(label, expanded=True):
                filepath = generated_files_map.get(entity_key)
                filename = os.path.basename(filepath) if filepath else f"{entity_key.lower()}.tsv"
                data_exists = len(st.session_state.metadata.get(entity_key, [])) > 0
                
                if data_exists and filepath and os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        st.download_button(
                            label=f"Download {filename}",
                            data=f.read(),
                            file_name=filename,
                            mime="text/tab-separated-values",
                            key=f"dl_btn_{entity_key}"
                        )
                else:
                    st.button(f"Download {filename}", key=f"dl_btn_disabled_{entity_key}", disabled=True)

                st.markdown("---")

                entity_data = st.session_state.metadata.get(entity_key)
                if entity_data:
                    if entity_key in ["Investigator", "Related_Work"]:
                        for idx, item in enumerate(entity_data):
                            st.write(f"**{entity_key} {idx+1}:**")
                            for key, value in item.items():
                                st.write(f"  - {key}: {value}")
                    else: # Program, Dataset
                        for key, value in entity_data[0].items():
                            st.write(f"**{key}:** {value}")
                else:
                    st.warning(f"No {entity_key.lower()} information provided.")

        st.markdown("---")
        
        if st.button("➡️ Proceed to Phase 1", use_container_width=True, type="primary"):
            st.session_state.phase = 1
            st.rerun()

# ============================================================================
# PHASE 1: STRUCTURE MAPPING & ORGANIZATION
# ============================================================================
elif st.session_state.phase == 1:
    st.header("Phase 1: Structure Mapping & Organization")

    # Prepare script download link
    try:
        with open("tcia_data_inventory.py", "rb") as f:
            script_data = f.read()
        b64_script = base64.b64encode(script_data).decode()
        script_link = f'<a href="data:file/python;base64,{b64_script}" download="tcia_data_inventory.py">try out this script</a>'
    except Exception:
        script_link = "try out this script"

    st.markdown(f"""
    Upload your source data files and map your columns to the TCIA target entities.

    As a convenience, {script_link} to create some TSV inventories of your dataset directory that can be uploaded for harmonization in the following steps. To run it, you will need Python with pydicom and pandas installed. After executing the script you will be asked what directory you'd like to analyze.
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your source data file (CSV, TSV, or Excel)",
        type=['csv', 'tsv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        # Read the file
        try:
            df = None
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                # Excel file
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox("Select which sheet to process:", sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    df = pd.read_excel(uploaded_file)
            
            if df is not None:
                # --- Basic Data Cleaning ---
                # 1. Drop completely empty rows and columns
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                # 2. Check for potential title rows (heuristic: first row has mostly NaNs)
                # If the first non-empty row has fewer non-NaN values than the next row, 
                # it might be a title row.
                if len(df) > 1:
                    first_row_non_nans = df.iloc[0].count()
                    second_row_non_nans = df.iloc[1].count()
                    if first_row_non_nans == 1 and second_row_non_nans > 1:
                        st.info("💡 Detected a potential title row. Using the next row as header.")
                        new_header = df.iloc[1]
                        df = df[2:]
                        df.columns = new_header
                
                # 3. Strip whitespace from headers and string values
                df.columns = [str(c).strip() for c in df.columns]
                df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

                st.session_state.uploaded_data = df
                st.success(f"✅ Loaded data with {len(df)} rows and {len(df.columns)} columns")
                
                # Show preview
                with st.expander("Preview Data", expanded=True):
                    st.dataframe(df.head(10))

            # ---------------- Master Data Dictionary Section ----------------
            st.markdown("### 🗂️ Generate Master Data Dictionary Template")
            st.markdown("Upload any spreadsheets (TSV/CSV/XLS/XLSX) to create a master Data Dictionary template.")

            uploaded_files = st.file_uploader(
                "Upload multiple files for Master Data Dictionary",
                accept_multiple_files=True,
                type=['csv', 'tsv', 'xlsx', 'xls']
            )

            if uploaded_files:
                dict_rows = []

                for f in uploaded_files:
                    try:
                        temp_fp = os.path.join(st.session_state.output_dir, f.name)
                        with open(temp_fp, 'wb') as out_file:
                            out_file.write(f.read())

                        # Load file
                        if f.name.endswith('.csv'):
                            df_dict = pd.read_csv(temp_fp)
                        elif f.name.endswith('.tsv'):
                            df_dict = pd.read_csv(temp_fp, sep='\t')
                        else:  # Excel
                            df_dict = pd.read_excel(temp_fp)

                        df_dict = df_dict.dropna(how='all').dropna(axis=1, how='all')
                        df_dict.columns = [str(c).strip() for c in df_dict.columns]

                        # Collect columns
                        for col in df_dict.columns:
                            unique_vals = df_dict[col].dropna().unique()
                            summary = ", ".join(map(str, unique_vals[:10]))
                            if len(unique_vals) > 10:
                                summary += ", ..."
                            dict_rows.append({
                                "Data Element Name": f"{f.name}.{col}",
                                "Data Element Values": summary,
                                "Common Data Element ID": "",
                                "Data Element Definition": ""
                            })
                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")

                if dict_rows:
                    dict_df = pd.DataFrame(dict_rows)
                    dict_path = os.path.join(st.session_state.output_dir, "master_data_dictionary.tsv")
                    dict_df.to_csv(dict_path, sep='\t', index=False)

                    with open(dict_path, 'r') as f:
                        st.download_button(
                            "Download Master Data Dictionary",
                            data=f.read(),
                            file_name="master_data_dictionary.tsv",
                            mime="text/tab-separated-values"
                        )
                    st.success("✅ Master Data Dictionary generated!")
            
            st.markdown("---")
            st.subheader("Column Mapping")
            st.markdown("Map your source columns to the TCIA target properties.")
            
            # Get all target properties from schema
            all_properties = {}
            excluded_entities = ['Program', 'Dataset', 'Investigator', 'Related_Work']
            phase0_linkages = ['program.', 'dataset.', 'investigator.', 'related_work.']

            for entity_name, properties in schema.items():
                if entity_name in excluded_entities:
                    continue
                for prop in properties:
                    prop_name = prop['Property']
                    # Exclude linkage properties that refer to Phase 0 entities
                    is_phase0_linkage = any(prop_name.startswith(prefix) for prefix in phase0_linkages)
                    if not is_phase0_linkage:
                        all_properties[f"{entity_name}.{prop_name}"] = prop
            
            # Create mapping interface
            st.write("**Map Source Columns to Target Properties:**")
            
            mapping_data = []
            for col in df.columns:
                cols = st.columns([3, 4, 2])
                with cols[0]:
                    st.write(f"**{col}**")
                    # Show sample values
                    sample_vals = df[col].dropna().unique()[:3]
                    st.caption(f"Sample: {', '.join(map(str, sample_vals))}")
                
                with cols[1]:
                    # Get current mapping if exists
                    current_mapping = None
                    for target, source in st.session_state.column_mapping.items():
                        if source == col:
                            current_mapping = target
                            break
                    
                    # Property selector
                    property_options = ["(Skip this column)"] + list(all_properties.keys())
                    default_index = 0
                    if current_mapping and current_mapping in property_options:
                        default_index = property_options.index(current_mapping)
                    
                    selected = st.selectbox(
                        "Target Property",
                        options=property_options,
                        index=default_index,
                        key=f"map_{col}",
                        label_visibility="collapsed"
                    )
                    
                    if selected != "(Skip this column)":
                        mapping_data.append((selected, col))
                
                with cols[2]:
                    if selected != "(Skip this column)" and selected in all_properties:
                        prop_info = all_properties[selected]
                        if prop_info.get('Required/optional') == 'R':
                            st.write("✅ Required")
                        else:
                            st.write("⚪ Optional")
            
            st.markdown("---")
            
            if st.button("✅ Confirm Mapping", type="primary"):
                # Save mapping
                st.session_state.column_mapping = {target: source for target, source in mapping_data}
                st.session_state.structure_approved = True
                st.success("✅ Column mapping confirmed!")
                st.info("Proceeding to Phase 2: Value Standardization...")
                
                # Check for conflicts with Phase 0 metadata
                conflicts = check_metadata_conflict(st.session_state.metadata, df, st.session_state.column_mapping)
                if conflicts:
                    st.warning("⚠️ Detected conflicts between uploaded data and Phase 0 metadata:")
                    for conflict in conflicts:
                        st.write(f"- {conflict['entity']}.{conflict['property']}: Initial='{conflict['initial_value']}' vs New='{conflict['new_value']}'")
                    st.write("Please review and update either your Phase 0 metadata or your uploaded data.")
            
            # Show proceed button if mapping is approved
            if st.session_state.structure_approved:
                st.markdown("---")
                if st.button("➡️ Proceed to Phase 2", type="primary", use_container_width=True):
                    st.session_state.phase = 2
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("👆 Please upload a file to begin structure mapping.")

# ============================================================================
# PHASE 2: VALUE STANDARDIZATION
# ============================================================================
elif st.session_state.phase == 2:
    st.header("Phase 2: Value Standardization")
    st.markdown("""
    Now let's standardize your data values to match TCIA permissible values using ontology-enhanced matching.
    """)
    
    if st.session_state.uploaded_data is None:
        st.warning("No data uploaded. Please go back to Phase 1.")
    elif not st.session_state.structure_approved:
        st.warning("Structure mapping not confirmed. Please complete Phase 1 first.")
    else:
        df = st.session_state.uploaded_data
        
        # Split data by schema
        split_data = split_data_by_schema(df, st.session_state.column_mapping, schema)
        
        st.write(f"**Identified {len(split_data)} target entities from your data.**")
        
        for entity_name, entity_df in split_data.items():
            with st.expander(f"📊 {entity_name} ({len(entity_df)} rows)", expanded=True):
                st.dataframe(entity_df.head(10))
                
                # Validate values
                report, corrections = validate_dataframe(entity_df, entity_name, schema, permissible_values)
                
                if report:
                    st.write("**Validation Issues Found:**")
                    for item in report[:10]:  # Show first 10
                        st.write(f"- {item}")
                    
                    if len(report) > 10:
                        st.info(f"... and {len(report) - 10} more issues")
                    
                    if corrections:
                        st.write("**Suggested Corrections:**")
                        for col, col_corrections in corrections.items():
                            st.write(f"Column: **{col}**")
                            for old_val, new_val in list(col_corrections.items())[:5]:
                                extra_info = ""
                                if col in permissible_values:
                                    matches = permissible_values[col]
                                    if matches and isinstance(matches[0], dict):
                                        match = next((m for m in matches if m['value'] == new_val), None)
                                        if match:
                                            parts = []
                                            if 'code' in match and match['code']:
                                                parts.append(f"Code: {match['code']}")
                                            if 'definition' in match and match['definition']:
                                                # Truncate definition if too long
                                                defn = match['definition']
                                                if len(defn) > 100:
                                                    defn = defn[:97] + "..."
                                                parts.append(f"Def: {defn}")
                                            if parts:
                                                extra_info = " (" + " | ".join(parts) + ")"
                                
                                st.write(f"  - '{old_val}' → **{new_val}**{extra_info}")
                        
                        if st.button(f"Apply Corrections to {entity_name}", key=f"apply_{entity_name}"):
                            # Apply corrections
                            for col, col_corrections in corrections.items():
                                entity_df[col] = entity_df[col].replace(col_corrections)
                            st.success(f"✅ Applied corrections to {entity_name}")
                            
                            # Save the corrected dataframe
                            output_file = os.path.join(st.session_state.output_dir, f"{entity_name.lower()}.tsv")
                            entity_df.to_csv(output_file, sep='\t', index=False)
                            st.success(f"✅ Saved to {output_file}")
                            
                            # Offer download
                            with open(output_file, 'r') as f:
                                st.download_button(
                                    label=f"Download {entity_name}.tsv",
                                    data=f.read(),
                                    file_name=f"{entity_name.lower()}.tsv",
                                    mime="text/tab-separated-values",
                                    key=f"download_{entity_name}"
                                )
                else:
                    st.success("✅ All values are valid!")
                    
                    # Save the dataframe
                    output_file = os.path.join(st.session_state.output_dir, f"{entity_name.lower()}.tsv")
                    entity_df.to_csv(output_file, sep='\t', index=False)
                    st.success(f"✅ Saved to {output_file}")
                    
                    # Offer download
                    with open(output_file, 'r') as f:
                        st.download_button(
                            label=f"Download {entity_name}.tsv",
                            data=f.read(),
                            file_name=f"{entity_name.lower()}.tsv",
                            mime="text/tab-separated-values",
                            key=f"download_{entity_name}"
                        )
        
        st.markdown("---")
        st.success("🎉 Remapping complete! All TSV files have been generated.")
        
        if st.button("🔄 Start New Remapping"):
            reset_app()
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
TCIA Dataset Remapper | Following TCIA Imaging Submission Data Model<br>
Leveraging NCIt, UBERON, and SNOMED ontologies for standardization
</div>
""", unsafe_allow_html=True)
