# ===== train.py (FINAL CLEAN) =====

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
OUTPUT_DIR = "./trained_router"

# ===== LOAD DATA =====
dataset = load_dataset("yahma/alpaca-cleaned")
dataset = dataset["train"].shuffle(seed=42).select(range(5000))

# ===== LABEL FUNCTION =====
def label_function(example):
    text = example["instruction"].lower()

    if len(text.split()) <= 4:
        label = 0
    elif "define" in text or "what is" in text:
        label = 1
    elif "explain" in text or "difference" in text:
        label = 2
    elif "how" in text or "working" in text:
        label = 3
    else:
        label = 4

    return {"label": label}

dataset = dataset.map(label_function)

# ===== TOKENIZE =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["instruction"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize)
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.remove_columns(["instruction", "input", "output"])

# ===== MODEL =====
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# ===== METRICS =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": (preds == labels).mean()}

# ===== TRAINING =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ===== TRAIN =====
trainer.train()

# ===== SAVE =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Model saved to:", OUTPUT_DIR)