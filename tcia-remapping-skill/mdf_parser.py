import yaml
import os

def load_mdf_model(model_path, props_path, terms_path):
    with open(model_path, 'r') as f:
        model = yaml.safe_load(f)
    with open(props_path, 'r') as f:
        props_defs = yaml.safe_load(f)
    with open(terms_path, 'r') as f:
        terms = yaml.safe_load(f)
    
    return model, props_defs, terms

def transform_mdf_to_schema(model, props_defs, terms):
    schema = {}
    permissible_values = {}
    relationships = model.get('Relationships', {})
    
    prop_definitions = props_defs.get('PropDefinitions', {})
    term_definitions = terms.get('Terms', {})
    
    for node_name, node_info in model.get('Nodes', {}).items():
        node_props = []
        for prop_name in node_info.get('Props', []):
            prop_def = prop_definitions.get(prop_name, {})
            
            # Map Req: 'Yes'/'No' to 'R'/'O'
            req = prop_def.get('Req')
            req_status = 'O'
            if req == 'Yes' or req is True:
                req_status = 'R'
            elif req == 'No' or req is False:
                req_status = 'O'
            
            node_props.append({
                'Property': prop_name,
                'Description': prop_def.get('Desc'),
                'Required/optional': req_status
            })
            
            # Extract permissible values (Enums)
            if 'Enum' in prop_def:
                enum_values = []
                for val in prop_def['Enum']:
                    val_info = {'value': val}
                    # Look up in terms for richer info
                    term_info = term_definitions.get(val, {})
                    if term_info:
                        if 'Code' in term_info:
                            val_info['code'] = term_info['Code']
                        if 'Definition' in term_info:
                            val_info['definition'] = term_info['Definition']
                        if 'Origin' in term_info:
                            val_info['origin'] = term_info['Origin']
                    
                    enum_values.append(val_info)
                
                permissible_values[prop_name] = enum_values
                
        schema[node_name] = node_props
        
    return schema, permissible_values, relationships

def get_mdf_resources(resource_dir):
    model_path = os.path.join(resource_dir, 'model', 'nci_imaging_submission_model.yml')
    props_path = os.path.join(resource_dir, 'model', 'nci_imaging_submission_model_properties.yml')
    terms_path = os.path.join(resource_dir, 'model', 'nci_imaging_submission_model_terms.yml')
    
    if all(os.path.exists(p) for p in [model_path, props_path, terms_path]):
        model, props, terms = load_mdf_model(model_path, props_path, terms_path)
        return transform_mdf_to_schema(model, props, terms)
    return None, None, None
