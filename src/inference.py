import os
import sys
import io

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dataset import MODEL_NAME

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class EnglishToHindiTranslator:
    def __init__(self, model_path="./model_output"):
        if os.path.exists(model_path):
            print(f"[INFO] Loading fine-tuned model from '{model_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            print(f"[WARNING] Fine-tuned checkpoint not found at '{model_path}'.")
            print(f"[INFO] Loading base pre-trained model '{MODEL_NAME}' instead...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def translate(self, text: str) -> str:
        if not text.strip():
            return "Please provide a valid sentence."

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text


def main():
    translator = EnglishToHindiTranslator()

    print("\n==================================================")
    print("  English to Hindi Transformer Translator CLI  ")
    print("==================================================")
    print("Type 'exit' or 'quit' to stop.\n")

    sample_sentences = [
        "Welcome to our college project demonstration.",
        "Artificial Intelligence will transform education.",
        "How are you doing today?",
    ]

    print("--- Running Sample Inferences ---")
    for sample in sample_sentences:
        hindi = translator.translate(sample)
        print(f"EN: {sample}")
        print(f"HI: {hindi}\n")

    print("--- Interactive Mode ---")
    while True:
        try:
            user_input = input("Enter English Text > ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting translator. Goodbye!")
                break
            translation = translator.translate(user_input)
            print(f"Hindi Translation: {translation}\n")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
