import os
import sys
import io

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"


def _repair_mojibake(text):
    """Repair UTF-8 Hindi text that was accidentally decoded as Latin-1."""
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def load_en_hi_dataset(csv_path="data/sample_en_hi_20k.csv", use_hf_dataset=False):
    if use_hf_dataset:
        print("[INFO] Loading dataset 'Helsinki-NLP/opus-100' (en-hi) from Hugging Face...")
        try:
            hf_ds = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train[:2500]")
            en_texts = [item["translation"]["en"] for item in hf_ds]
            hi_texts = [item["translation"]["hi"] for item in hf_ds]
            df = pd.DataFrame({"english": en_texts, "hindi": hi_texts})
        except Exception as e:
            print(f"[WARNING] Could not load Hugging Face dataset ({e}). Falling back to local CSV...")
            df = pd.read_csv(csv_path, encoding='utf-8')
    else:
        if not os.path.exists(csv_path) and os.path.exists("data/sample_en_hi.csv"):
            csv_path = "data/sample_en_hi.csv"

        print(f"[INFO] Loading dataset from local CSV: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8')

    df = df[["english", "hindi"]].dropna().reset_index(drop=True)
    df["hindi"] = df["hindi"].map(_repair_mojibake)
    print(f"[SUCCESS] Successfully loaded {len(df)} sentence pairs.")
    return df


def preprocess_and_tokenize(df, max_length=128, test_size=0.2):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_dataset = Dataset.from_pandas(df)

    if len(df) <= 5:
        test_size = 1 / len(df)

    split_dataset = hf_dataset.train_test_split(test_size=test_size, seed=42)

    def preprocess_function(examples):
        inputs = examples["english"]
        targets = examples["hindi"]
        
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        
        labels = tokenizer(
            text_target=targets,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("[INFO] Tokenizing dataset with SentencePiece...")
    tokenized_dataset = split_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["english", "hindi"],
    )

    return tokenized_dataset, tokenizer


if __name__ == "__main__":
    df = load_en_hi_dataset()
    tokenized_ds, _ = preprocess_and_tokenize(df)
    print("[RESULT] Train dataset size:", len(tokenized_ds["train"]))
    print("[RESULT] Test dataset size :", len(tokenized_ds["test"]))
