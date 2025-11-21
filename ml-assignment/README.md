#  Trigram Language Model – ML Internship Assessment

This directory contains the implementation of the **trigram-based language model** for the ML internship assignment.  
The solution includes:

-  Fully implemented `TrigramModel` class
-  Unit tests support (compatible with provided tests)
-  CLI-based demonstration script
-  Design explanation (`evaluation.md`)
-  Optional Task: Scaled Dot-Product Attention using **only NumPy**

##  Setup Instructions

### 1️⃣ Create & activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

##  How to Run Tests

```bash
cd ml-assignment
pytest tests/test_ngram.py
```

##  Run the Language Model Example

```bash
python src/generate.py
```

##  Project Structure

ml-assignment/
│── README.md
│── evaluation.md
│── data/
│   └── example_corpus.txt
│── src/
│   ├── ngram_model.py
│   └── generate.py
│── tests/
│   └── test_ngram.py
│── attention/
│   └── scaled_dot_product_attention.py

##  Design Choices Summary

| Component | Decision |
|----------|----------|
| Sentence splitting | regex over ., !, ? |
| Tokenization | lower-case + alphanumeric regex |
| Padding | <s> <s> ... </s> |
| Data structure | dict of dict |
| Generation | random sampling |
| Unknown tokens | <unk> |
| Stopping criteria | </s> or max_length |

See `evaluation.md` for details.

##  Optional Task

```bash
cd attention
python scaled_dot_product_attention.py
```

##  Dependencies

pytest
numpy

##  Suggested Commit Message

git commit -m "Implemented trigram language model and optional attention task with documentation"
