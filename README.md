# English to Hindi Transformer-Based Machine Translation

## Overview

This project implements a Transformer-based Neural Machine Translation (NMT) system that translates text from English to Hindi. The model is fine-tuned on a bilingual parallel corpus to generate accurate Hindi translations while preserving the meaning of the input sentence. The project demonstrates the complete machine translation pipeline, including data preprocessing, tokenization, model training, evaluation, and inference.

## Features

- English to Hindi text translation
- Transformer-based architecture using Hugging Face Transformers
- Tokenization using a pretrained tokenizer
- Fine-tuning on bilingual parallel datasets
- Evaluation using BLEU score
- Interactive inference for translating custom sentences
- Modular and beginner-friendly project structure

## Technology Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- SentencePiece
- NumPy
- Pandas
- Scikit-learn
- Evaluate
- Matplotlib
- VS Code

## Project Structure

```text
english_to_hindi_transformer/
│
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── models/
│
├── checkpoints/
│
├── train.py
├── inference.py
├── preprocess.py
├── evaluate_model.py
├── utils.py
├── requirements.txt
├── README.md
└── outputs/
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd english_to_hindi_transformer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses an English-Hindi parallel corpus containing aligned sentence pairs.

Example:

| English | Hindi |
| --- | --- |
| Good Morning | सुप्रभात |
| How are you? | आप कैसे हैं? |
| I love machine learning. | मुझे मशीन लर्निंग पसंद है। |

The dataset is divided into:

- Training Set
- Validation Set
- Test Set

## Training

Train the model using:

```bash
python train.py
```

During training the model:

- Loads the dataset
- Tokenizes English and Hindi sentences
- Fine-tunes the Transformer model
- Saves checkpoints after each epoch
- Stores the best-performing model

## Model Evaluation

Evaluate the trained model using:

```bash
python evaluate_model.py
```

Evaluation metrics include:

- BLEU Score
- Validation Loss

## Inference

Translate custom English sentences:

```bash
python inference.py
```

Example:

Input

```text
The weather is pleasant today.
```

Output

```text
आज मौसम सुहावना है।
```

## Applications

- Language translation
- Educational tools
- Government document translation
- Digital communication
- AI-based language learning
- Research in Natural Language Processing

## Future Improvements

- Support multiple Indian languages
- Deploy using FastAPI
- Build a web interface with Streamlit
- Improve translation accuracy with larger datasets
- Add speech-to-text and text-to-speech capabilities

## Learning Outcomes

Through this project, the following concepts were explored:

- Neural Machine Translation
- Transformer architecture
- Tokenization
- Transfer learning
- Fine-tuning pretrained models
- Model evaluation
- Natural Language Processing using PyTorch and Hugging Face

## Conclusion

This project demonstrates a complete English-to-Hindi Neural Machine Translation system using Transformer models. It provides practical experience with modern NLP techniques, including data preprocessing, model training, evaluation, and inference. The implementation serves as a strong foundation for understanding Transformer-based translation systems and can be extended to support additional languages and advanced NLP applications.

## Author

**Ankur Singh**

B.Tech CSE (AI & ML)

GLA University, Mathura
