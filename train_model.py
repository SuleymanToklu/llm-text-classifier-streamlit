import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset

print("Loading processed data...")
df = pd.read_csv('processed_train.csv')

df = df.sample(n=2000, random_state=42)
print(f"Using a subset of {len(df)} samples for this run.")

model_name = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)


df = df.rename(columns={'winner': 'labels'})

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_function(examples):
    return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["input_text", "__index_level_0__"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["input_text", "__index_level_0__"])


training_args = TrainingArguments(
    output_dir="./results",            
    num_train_epochs=2,                  
    per_device_train_batch_size=4,               
    warmup_steps=500,         
    weight_decay=0.01,                   
    logging_dir="./logs",                
    logging_steps=10,        
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

print("Starting model training...")
trainer.train()

final_model_path = "./llm_winner_classifier"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Training complete. Best model saved to {final_model_path}")