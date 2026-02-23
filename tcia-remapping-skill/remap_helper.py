import pandas as pd
import json
import difflib
import os

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_closest_match(value, choices, n=1, cutoff=0.6):
    """
    choices can be a list of strings or a list of dicts with a 'value' key.
    """
    if not value or pd.isna(value):
        return None

    # Extract string values if choices is a list of dicts
    if choices and isinstance(choices[0], dict):
        choice_strings = [c['value'] for c in choices]
        value_to_choice = {c['value']: c for c in choices}
    else:
        choice_strings = choices
        value_to_choice = {c: c for c in choices}

    matches = difflib.get_close_matches(str(value), choice_strings, n=n, cutoff=cutoff)
    if matches:
        return value_to_choice[matches[0]]
    return None

def validate_dataframe(df, schema_name, schema, permissible_values):
    """
    Returns a summary report and a dict of suggested corrections.
    """
    report = []
    corrections = {}

    target_props = {p['Property']: p for p in schema.get(schema_name, [])}

    for col in df.columns:
        if col in target_props:
            if col in permissible_values:
                choices = permissible_values[col]
                # If choices is list of dicts, get values for easy lookup
                if choices and isinstance(choices[0], dict):
                    valid_set = {c['value'] for c in choices}
                else:
                    valid_set = set(choices)

                invalid_values = []
                for val in df[col].unique():
                    if pd.isna(val) or str(val).strip() == "":
                        continue
                    if val not in valid_set:
                        invalid_values.append(val)

                if invalid_values:
                    col_corrections = {}
                    for val in invalid_values:
                        match = get_closest_match(val, choices)
                        if match:
                            # match is either a string or a dict
                            suggestion = match['value'] if isinstance(match, dict) else match
                            col_corrections[val] = suggestion
                            report.append(f"Column '{col}': '{val}' -> Suggested: '{suggestion}'")
                        else:
                            report.append(f"Column '{col}': '{val}' -> No close match found.")
                    if col_corrections:
                        corrections[col] = col_corrections

    return report, corrections

def split_data_by_schema(df, column_mapping, schema):
    """
    Split source dataframe into target entities based on column mapping.
    column_mapping can be in format:
    - {"property": "source_col"} or
    - {"Entity.property": "source_col"}
    """
    results = {}
    for sheet_name, properties in schema.items():
        target_cols = [p['Property'] for p in properties]
        new_df = pd.DataFrame()
        
        for prop in target_cols:
            # Check both formats: "property" and "Entity.property"
            source_col = None
            if prop in column_mapping:
                source_col = column_mapping[prop]
            elif f"{sheet_name}.{prop}" in column_mapping:
                source_col = column_mapping[f"{sheet_name}.{prop}"]
            
            if source_col and source_col in df.columns:
                new_df[prop] = df[source_col]
        
        if not new_df.empty:
            new_df = new_df.drop_duplicates()
            results[sheet_name] = new_df
    
    return results

def write_metadata_tsv(entity_name, data, schema, output_dir='.'):
    """
    data: list of dicts or a single dict
    schema: the full schema dict
    """
    if isinstance(data, dict):
        data = [data]

    properties = [p['Property'] for p in schema.get(entity_name, [])]
    if not properties:
        return None

    df = pd.DataFrame(data)
    # Ensure all columns exist and are in order
    for prop in properties:
        if prop not in df.columns:
            df[prop] = None

    df = df[properties]
    filename = f"{entity_name.lower()}.tsv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, sep='\t', index=False)
    return filepath

def check_metadata_conflict(initial_metadata, df, column_mapping):
    """
    Checks if values in df (after mapping) conflict with initial_metadata.
    initial_metadata: dict of {entity_name: [list_of_dicts]}
    """
    conflicts = []
    for entity_name, meta_list in initial_metadata.items():
        if not meta_list:
            continue

        # We focus on Program and Dataset which are usually singular per submission
        if entity_name not in ["Program", "Dataset"]:
            continue

        meta_dict = meta_list[0] # Assume one for now

        for target_col, source_col in column_mapping.items():
            if target_col in meta_dict:
                meta_val = str(meta_dict[target_col]).strip()
                if source_col in df.columns:
                    unique_vals = df[source_col].dropna().unique()
                    for val in unique_vals:
                        if str(val).strip() != meta_val:
                            conflicts.append({
                                'entity': entity_name,
                                'property': target_col,
                                'initial_value': meta_val,
                                'new_value': val
                            })
    return conflicts

def check_missing_links(split_data, schema, relationships):
    """
    Checks if required linkages (based on relationships) are present in the split data.
    """
    missing_links = []
    for entity_name, entity_df in split_data.items():
        for rel_name, rel_info in relationships.items():
            for end in rel_info.get('Ends', []):
                if end['Src'] == entity_name:
                    dst_lower = end['Dst'].lower()
                    linkage_prop = next((p['Property'] for p in schema.get(entity_name, []) if p['Property'].startswith(f"{dst_lower}.")), None)
                    if linkage_prop and linkage_prop not in entity_df.columns:
                        # Only report if it's not a link to a Phase 0 entity (which we handle automatically)
                        # Actually, better to report it and let the caller decide.
                        missing_links.append({
                            'entity': entity_name,
                            'target_entity': end['Dst'],
                            'property': linkage_prop
                        })
    return missing_links

def main():
    print("TCIA Remap Helper loaded with metadata support.")

if __name__ == "__main__":
    main()
