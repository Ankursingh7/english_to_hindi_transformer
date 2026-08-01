# 🎓 College Viva & Presentation Preparation Guide

## 1. Project Pitch (30-Second Elevator Pitch)
> "Good morning/afternoon Professors. Our project is an **English to Hindi Neural Machine Translation System** powered by a fine-tuned Transformer model (`MarianMT / Helsinki-NLP`). Instead of relying on statistical or rule-based translation, we leverage a Sequence-to-Sequence (Seq2Seq) Transformer architecture with self-attention mechanisms to generate fluent and contextually accurate Hindi translations. We tokenized parallel sentence pairs using SentencePiece, fine-tuned the model using PyTorch and Hugging Face Transformers, and evaluated its translation quality using BLEU scores."

---

## 2. Frequently Asked Viva Questions & Standard Answers

### Q1: What is a Transformer model, and why is it better than RNNs or LSTMs for translation?
**Answer**: 
- **Sequential Bottleneck**: RNNs and LSTMs process words sequentially (one word at a time), which makes training slow and struggles with long-range dependencies.
- **Self-Attention**: Transformers use **Multi-Head Self-Attention** to process all tokens in parallel, enabling fast GPU compute and capturing relationships between distant words in a sentence regardless of distance.

### Q2: Why did you choose Fine-Tuning instead of training a model from scratch?
**Answer**:
- Training a Transformer from scratch requires millions of sentence pairs, multi-GPU clusters, and weeks of compute.
- Fine-tuning a pre-trained model (like `MarianMT` trained on OPUS data) leverages **Transfer Learning**. The model already understands English-Hindi grammar and vocabulary; fine-tuning specializes or adapts it on our dataset efficiently with minimal compute in minutes.

### Q3: What is the role of Tokenization and SentencePiece?
**Answer**:
- Tokenization converts raw text strings into numerical IDs that neural networks understand.
- **SentencePiece** is a subword tokenization algorithm (like Byte-Pair Encoding). It splits out-of-vocabulary or complex morphologically rich Hindi words into subword units (e.g., "प्रोग्रामिंग" -> "प्रोग्राम" + "िंग"), preventing `<UNK>` (unknown token) issues.

### Q4: How is BLEU Score calculated, and what does it measure?
**Answer**:
- **BLEU (Bilingual Evaluation Understudy)** measures N-gram precision overlap between the model's generated candidate translation and ground-truth reference human translations.
- It calculates modified n-gram precision (1-gram up to 4-gram) with a brevity penalty to punish translations that are too short. Scores range from 0 to 100, where higher scores indicate higher translation fidelity.

### Q5: What is the Encoder-Decoder architecture in Seq2Seq?
**Answer**:
- **Encoder**: Converts input English sentence tokens into contextual dense vector embeddings representations.
- **Decoder**: Takes encoder representation and auto-regressively predicts Hindi output tokens one by one using cross-attention over encoder output.

### Q6: What decoding strategy did you use during inference?
**Answer**:
- We used **Beam Search Decoding** (with `num_beams=4`). Instead of greedy decoding (choosing only the highest probability single token at each step), Beam Search maintains top 4 candidate sequences at each step, returning the sequence with highest cumulative probability.

---

## 3. Key Concepts Checklist for Demonstration
- [x] Show `train.py` output showing loss decreasing epoch by epoch.
- [x] Show `evaluate.py` output displaying computed BLEU score.
- [x] Run `inference.py` live to translate custom English input entered by examiners.
