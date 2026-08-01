# 🚀 Training 20,000 Sentence Pairs on Google Colab (Free GPU)

Training 20,000 sentence pairs on a local laptop CPU takes ~3 to 4 hours, but on Google Colab's free T4 GPU, it takes **under 3 minutes**!

---

## 📋 Steps to Run on Google Colab

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a **New Notebook**.

### 2. Enable GPU Acceleration
Click on **Runtime > Change runtime type**, select **T4 GPU**, and click **Save**.

### 3. Copy & Run the Code in Colab

```python
# Cell 1: Install Dependencies
!pip install -q torch transformers datasets evaluate sacrebleu sentencepiece accelerate sacremoses pandas

# Cell 2: Download/Generate 20k Dataset & Fine-Tune MarianMT
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

print("📥 Loading Helsinki-NLP/opus-100 dataset (20,000 sentence pairs)...")
raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train[:20000]")

en_texts = [item["translation"]["en"] for item in raw_dataset]
hi_texts = [item["translation"]["hi"] for item in raw_dataset]
df = pd.DataFrame({"english": en_texts, "hindi": hi_texts})

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_dataset = Dataset.from_pandas(df)
split_ds = hf_dataset.train_test_split(test_size=0.1, seed=42)

def preprocess_fn(examples):
    inputs = tokenizer(examples["english"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["hindi"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

print("⚡ Tokenizing dataset...")
tokenized_ds = split_ds.map(preprocess_fn, batched=True, remove_columns=["english", "hindi"])

# Model Setup
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device.upper()}")
model.to(device)

training_args = Seq2SeqTrainingArguments(
    output_dir="./model_output_20k",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_total_limit=1,
    predict_with_generate=True,
    fp16=True, # Fast Mixed Precision on GPU
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

print("🚀 Starting GPU Fine-Tuning...")
trainer.train()

model.save_pretrained("./model_output_20k")
tokenizer.save_pretrained("./model_output_20k")
print("🎉 Model successfully fine-tuned on 20,000 sentence pairs!")
```
