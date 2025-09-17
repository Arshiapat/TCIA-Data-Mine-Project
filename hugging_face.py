import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import pipeline


# Example: Export form responses as CSV from Google Sheets
df = pd.read_csv("form_responses.csv")

print(df.head())


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

'''
with open("student_stress_survey.metadata.json", "w") as f:
    json.dump(croissant_metadata, f, indent=2)
'''




dataset = Dataset.from_pandas(df)
print(dataset)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["What is one sentence about how you feel this semester?"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(tokenized_dataset[0])

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # could also split into train/test
    eval_dataset=tokenized_dataset,
)

trainer.train()

# Use a sentiment analysis model on free-response text
classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

df["sentiment"] = df["What is one sentence about how you feel this semester?"].apply(lambda x: classifier(x)[0]['label'])
print(df[["What is one sentence about how you feel this semester?", "sentiment"]])
