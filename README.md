# 🧠 Mental Health Disorder Classifier

A fine-tuned **DistilBERT** model that classifies Reddit posts into 6 mental 
health categories. Built as part of a thesis comparing deep learning approaches 
for mental health NLP.

## 🎯 What it does
Enter a text description of how you're feeling, and the model predicts which 
of 6 mental health conditions the text most resembles:
**ADHD · Anxiety · Bipolar · Depression · PTSD · None**

## 🔬 Models Compared
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| BiLSTM (baseline) | 75.18% | 75.47% |
| DistilBERT (deployed) | 81.66% | 81.66% |
| RoBERTa | 85.33% | 85.00% |

Our models outperform previous state-of-the-art (AlBERT: 80.45% F1).

## 📦 Dataset
- **Mental Health Reddit Dataset** — 16,931 posts
- Train/Dev/Test: 13,727 / 1,488 / 1,488
- Source: Reddit mental health communities

## 🛠️ Tech Stack
PyTorch · HuggingFace Transformers · DistilBERT · FastAPI · Docker · Tailwind CSS

## ⚠️ Disclaimer
This tool is for research purposes only and is not a clinical diagnostic tool.
