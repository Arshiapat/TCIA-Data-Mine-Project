'''
This file contains a process to manually create JSON files with croissant guidelines.
'''

# IMPORTS
import pandas as pd
import json

# Creation of a pandas dataframe with example csv file
df = pd.read_csv("example_student_stress.csv")


# croissant's labels for JSON file
croissant_metadata = {
    "@context": "https://mlcommons.org/croissant/context/v1",
    "name": "student_stress_survey",
    "description": "Survey responses collected from a Google Form about study habits and stress.",
    "license": "CC-BY",
    "features": [
        {"name": "study_hours", "type": "integer"},
        {"name": "stress_level", "type": "integer"},
        {"name": "free_response", "type": "string"}
    ]
}

# Creation of croissant JSON file
with open("student_stress_survey.metadata.json", "w") as f:
    json.dump(croissant_metadata, f, indent=2)
