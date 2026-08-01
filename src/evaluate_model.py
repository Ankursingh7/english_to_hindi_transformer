import os
import sys
import io

import pandas as pd
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dataset import MODEL_NAME, load_en_hi_dataset

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def evaluate_bleu(model_path="./model_output", csv_path="data/sample_en_hi_large.csv"):
    if os.path.exists(model_path):
        print(f"[INFO] Loading fine-tuned model from '{model_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        print(f"[WARNING] Fine-tuned model not found at '{model_path}'. Using base pre-trained model '{MODEL_NAME}'.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    df = load_en_hi_dataset(csv_path=csv_path)
    sources = df["english"].tolist()[:100]
    references = [df["hindi"].tolist()[:100]]

    print(f"[INFO] Generating translations for {len(sources)} sentences for BLEU evaluation...")
    predictions = []

    model.eval()
    with torch.no_grad():
        for src in sources:
            inputs = tokenizer(
                src,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(translated)

    bleu = sacrebleu.corpus_bleu(predictions, references)

    print("\n---------------- EVALUATION RESULTS ----------------")
    print(f"BLEU Score: {bleu.score:.2f} / 100")
    print(f"Sample Source:     {sources[0]}")
    print(f"Target Reference: {references[0][0]}")
    print(f"Model Prediction: {predictions[0]}")
    print("----------------------------------------------------\n")

    return bleu.score


if __name__ == "__main__":
    evaluate_bleu()
