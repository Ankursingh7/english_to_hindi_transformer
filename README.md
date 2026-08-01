# English to Hindi Neural Machine Translation using Hugging Face Transformers

A beginner-friendly B.Tech CSE (AI & ML) project to fine-tune a Transformer-based Seq2Seq model (`Helsinki-NLP/opus-mt-en-hi`) for translating English text to Hindi.

---

## 📁 Project Structure

```
english_to_hindi_transformer/
├── data/
│   └── sample_en_hi.csv      # Sample dataset of parallel English-Hindi sentence pairs
├── src/
│   ├── dataset.py            # Data loading, cleaning, and tokenization logic
│   ├── train.py              # Fine-tuning loop using Hugging Face Seq2SeqTrainer
│   ├── evaluate.py           # Evaluates model performance using BLEU score
│   └── inference.py          # Interactive command-line translation tool
├── requirements.txt          # Python dependencies
├── README.md                 # Project instructions & overview
└── viva_prep.md              # College Viva questions, answers & presentation guide
```

---

## ⚡ Quick Start Guide (VS Code)

### Step 1: Open project in VS Code
Open VS Code and navigate to `File > Open Folder...` and select `english_to_hindi_transformer`.

### Step 2: Set up Virtual Environment
Open the terminal in VS Code (`Ctrl + ~`) and run:
```bash
python -m venv venv
venv\Scripts\activate      # On Windows PowerShell
# source venv/bin/activate  # On Linux/macOS
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Training
Fine-tune the pre-trained Helsinki-NLP model on the dataset:
```bash
python src/train.py
```

### Step 5: Evaluate the Model (BLEU Score)
```bash
python src/evaluate.py
```

### Step 6: Run Interactive Translation Inference
```bash
python src/inference.py
```

---

## 🎯 Model Architecture Overview
- **Model Base**: MarianMT (`Helsinki-NLP/opus-mt-en-hi`)
- **Architecture**: Transformer Encoder-Decoder (Seq2Seq)
- **Tokenization**: SentencePiece subword tokenization
- **Metric**: BLEU Score (via Hugging Face `evaluate` & `sacrebleu`)
