# 🧠 Gen-Z Slang Sentiment Analysis

> *Can a neural network understand "no cap, that was cooked"? Apparently yes — with 90% accuracy.*

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![F1 Score](https://img.shields.io/badge/F1%20Score-~90%25-brightgreen?style=flat-square)
![Model](https://img.shields.io/badge/Model-Bi--LSTM-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

---

## 📌 Overview

A deep learning NLP pipeline that classifies Gen-Z slang-heavy text into **5 sentiment categories** using a Bidirectional LSTM architecture. Built from scratch — including a hand-crafted dataset designed to reflect how Gen-Z actually writes online.

This project explores a genuine challenge in NLP: slang is **contextual, overlapping, and constantly evolving**. Words like *"no cap"* or *"ngl"* shift meaning entirely depending on context. The model learns to handle exactly that.

---

## 🎯 Sentiment Classes

| Label | Example |
|-------|---------|
| ✅ Positive | *"bro that food was bussin no cap"* |
| ❌ Negative | *"ngl that was actually so mid"* |
| 😂 Humorous | *"I'm dead 💀 he really said that"* |
| 😤 Frustration | *"this is so cooked I can't even rn"* |
| 😐 Neutral | *"no cap I have class at 8am"* |

---

## 🏗️ Project Architecture

```
gen-z-sentiment/
│
├── dataset/               # Custom hand-crafted slang dataset (200+ sentences/class)
├── GenZ.ipynb             # Full pipeline: preprocessing → training → evaluation
├── documentation/         # Methodology and design decisions
└── README.md
```

---

## ⚙️ Methodology

### 1. Dataset Design
Built entirely from scratch — **200+ sentences per class**, designed with:
- **Overlapping slang** across classes (e.g., *"no cap"* appears in Neutral AND Positive) to force contextual learning
- **Emoji-to-token conversion** (💀 → `emoji_skull`) to preserve sentiment cues
- Varied sentence structures: questions, exclamations, compound sentences

### 2. Preprocessing Pipeline
- Lowercase normalization + punctuation stripping
- Custom emoji dictionary → descriptive tokens
- Tokenization → integer sequences → zero-padding for uniform input length

### 3. Model: Bidirectional LSTM

```
Input → Embedding Layer → Bi-LSTM → Dropout → Dense (Softmax) → 5 Classes
```

**Why Bi-LSTM?**
- Reads sentences **both forward and backward** — critical for slang where meaning depends on full context
- Outperformed standard LSTM, CNN, and RNN architectures on this dataset
- Transformers were intentionally avoided (overkill for dataset size; goal was architectural understanding)

### 4. Training
- GPU-accelerated on **T4 GPU** via Google Colab
- EarlyStopping to prevent overfitting
- Class weights applied to handle any imbalance

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | ~90% |
| F1 Score (weighted) | ~0.90 |
| Best Classes | Positive, Negative, Frustration |

Evaluated using accuracy, per-class precision/recall/F1, and confusion matrix visualization.

---

## 🔑 Key Takeaways

- **Dataset quality > dataset size.** 200 thoughtfully constructed sentences outperformed naive data collection.
- **Emoji tokenization matters.** Converting 💀 to `emoji_skull` meaningfully improved the Humorous class.
- **Architecture choice is decisive.** Switching from LSTM → Bi-LSTM was the single biggest accuracy jump.
- **Embedding dimension tuning** helped the model differentiate subtle sentiment signals.

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/gen-z-sentiment-analysis.git

# Open the notebook
jupyter notebook GenZ.ipynb
# Or open directly in Google Colab
```

**Dependencies:** TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, Google Colab (T4 GPU recommended)

---

## 👩‍💻 Author

**Thanishka Thatikonda**
B.E. Mathematics and Computing, BITS Pilani
[LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) · [GitHub](https://github.com/YOUR_USERNAME)

---

*Self-initiated project, 2025. Built to explore NLP on non-standard, evolving language.*
