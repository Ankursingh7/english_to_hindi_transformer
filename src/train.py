import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataset import load_en_hi_dataset, preprocess_and_tokenize, MODEL_NAME


def train_model(
    csv_path="data/sample_en_hi_large.csv",
    use_hf_dataset=False,
    output_dir="./model_output",
    epochs=3,
    batch_size=8,
):
    df = load_en_hi_dataset(csv_path=csv_path, use_hf_dataset=use_hf_dataset)
    tokenized_dataset, tokenizer = preprocess_and_tokenize(df)

    print(f"[INFO] Loading pre-trained Transformer model: {MODEL_NAME}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device.upper()}")
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    try:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=epochs,
            predict_with_generate=True,
            logging_dir="./logs",
            logging_steps=20,
            report_to="none",
        )
    except TypeError:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=epochs,
            predict_with_generate=True,
            logging_dir="./logs",
            logging_steps=20,
            report_to="none",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    print("[INFO] Starting fine-tuning training loop on 2,000 sentence pairs...")
    trainer.train()

    print(f"[INFO] Saving fine-tuned model and tokenizer to '{output_dir}'...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[SUCCESS] Training complete! Model saved successfully.")


if __name__ == "__main__":
    train_model(epochs=3, batch_size=8)
