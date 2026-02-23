import requests
import re

def search_orcid_by_name(given_names=None, family_name=None, text=None):
    """Search ORCID by name and return a list of potential matches."""
    headers = {"Accept": "application/json"}
    if text:
        # Use simple text search
        query = text.replace(' ', '+')
    else:
        query_parts = []
        if given_names:
            query_parts.append(f"given-names:{given_names.strip()}")
        if family_name:
            query_parts.append(f"family-name:{family_name.strip()}")
        query = " AND ".join(query_parts)

    url = f"https://pub.orcid.org/v3.0/search/?q={query}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('result', [])
            return [r.get('orcid-identifier', {}).get('path') for r in results]
    except Exception:
        pass
    return []

def get_orcid_profile(orcid_id):
    """Fetch full profile details for an ORCID ID."""
    if not orcid_id:
        return None
    headers = {"Accept": "application/json"}
    try:
        # Get person details (names)
        url_p = f"https://pub.orcid.org/v3.0/{orcid_id}/person"
        res_p = requests.get(url_p, headers=headers, timeout=10)

        # Get employments (organization)
        url_e = f"https://pub.orcid.org/v3.0/{orcid_id}/employments"
        res_e = requests.get(url_e, headers=headers, timeout=10)

        profile = {
            'orcid_id': orcid_id,
            'given_names': '',
            'family_name': '',
            'organization': ''
        }

        if res_p.status_code == 200:
            data_p = res_p.json()
            name = data_p.get('name', {})
            if name:
                profile['given_names'] = name.get('given-names', {}).get('value', '') if name.get('given-names') else ''
                profile['family_name'] = name.get('family-name', {}).get('value', '') if name.get('family-name') else ''

        if res_e.status_code == 200:
            data_e = res_e.json()
            affiliations = data_e.get('affiliation-group', [])
            if affiliations:
                # Use the first one
                summaries = affiliations[0].get('summaries', [{}])
                if summaries:
                    emp = summaries[0].get('employment-summary', {})
                    org = emp.get('organization', {})
                    profile['organization'] = org.get('name', '')

        return profile
    except Exception:
        pass
    return None

def get_profiles_for_name(first_name, last_name):
    """Search for profiles by name and return detail list."""
    ids = search_orcid_by_name(given_names=first_name, family_name=last_name)
    profiles = []
    for orcid_id in ids[:5]: # Limit to top 5 matches
        profile = get_orcid_profile(orcid_id)
        if profile:
            profiles.append(profile)
    return profiles

def parse_author_input(text):
    """
    Attempts to parse a line of text into Name components or ORCID.
    Supports formats:
    - John Smith
    - Smith, John
    - 0000-0000-0000-0000
    - John Smith (0000-0000-0000-0000)
    """
    orcid_pattern = r'(\d{4}-\d{4}-\d{4}-\d{3}[\dX])'
    orcid_match = re.search(orcid_pattern, text)
    orcid = orcid_match.group(1) if orcid_match else None

    # Remove ORCID from text for name parsing
    name_part = re.sub(orcid_pattern, '', text).strip()
    name_part = name_part.replace('(', '').replace(')', '').strip()

    first_name = ""
    last_name = ""

    if name_part:
        if ',' in name_part:
            parts = name_part.split(',')
            last_name = parts[0].strip()
            first_name = parts[1].strip() if len(parts) > 1 else ""
        else:
            parts = name_part.split()
            if len(parts) >= 2:
                first_name = parts[0].strip()
                last_name = " ".join(parts[1:]).strip()
            else:
                first_name = parts[0].strip()

    return {
        'first_name': first_name,
        'last_name': last_name,
        'orcid': orcid,
        'original_text': text
    }
