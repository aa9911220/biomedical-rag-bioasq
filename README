# Biomedical Question Answering with Retrieval-Augmented Generation

## Overview

This project investigates how different retrieval strategies affect biomedical yes/no question answering with a small instruction-tuned language model.

Using a BioASQ-derived biomedical QA dataset, the project compares a retrieval-free baseline with several Retrieval-Augmented Generation (RAG) approaches.

The main research question is:

> **How does the quality and type of retrieved evidence affect biomedical question answering?**

## Models

### Generator

**Qwen/Qwen2.5-1.5B-Instruct**

The language model is responsible for generating the final `yes` or `no` answer.

### Dense Retriever

**sentence-transformers/all-MiniLM-L6-v2**

The model converts questions and candidate evidence sentences into embeddings. Cosine similarity is used to retrieve semantically relevant evidence.

## Experiments

### 1. Baseline

The question is directly provided to Qwen2.5-1.5B-Instruct without external retrieval.

**Question → Qwen → Yes / No**

### 2. Oracle Context RAG

The gold context provided by the dataset is supplied directly to Qwen2.5-1.5B-Instruct.

**Question + Gold Context → Qwen → Yes / No**

This experiment provides an approximate upper bound for retrieval-based approaches because the model receives the dataset's provided evidence.

### 3. BM25 Sentence-level RAG

BM25 is used to retrieve individual sentences from the training corpus.

**Question → BM25 → Top-k sentences → Qwen → Yes / No**

### 4. BM25 Document-level RAG

BM25 retrieval is performed at the document level. Retrieved documents are provided to the language model as evidence.

### 5. Dense RAG

Dense semantic retrieval is performed using `all-MiniLM-L6-v2`.

The retrieved evidence is then passed to Qwen2.5-1.5B-Instruct.

**Question → MiniLM → Semantic Retrieval → Top-k Evidence → Qwen → Yes / No**

Two retrieval settings were investigated:

- Top-3
- Top-5

## Results

| Method | Retrieval | Top-K | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Baseline | None | — | **0.743** | **0.722** |
| Oracle Context RAG | Gold context | — | **0.887** | **0.849** |
| BM25 Sentence RAG | BM25 | 3 | 0.370 | ~0.35 |
| BM25 Document RAG | BM25 | 3 | 0.420 | ~0.41 |
| Dense RAG | all-MiniLM-L6-v2 | 3 | 0.740 | 0.660 |
| Dense RAG | all-MiniLM-L6-v2 | 5 | **0.750** | **0.680** |

## Key Findings

### Oracle Context RAG

Oracle Context RAG achieved an Accuracy of **0.887** and a Macro-F1 of **0.849**, substantially outperforming the retrieval-free baseline.

This suggests that high-quality evidence can significantly improve biomedical question answering.

### BM25 Retrieval

BM25-based retrieval performed substantially worse than the baseline.

This suggests that lexical matching alone may retrieve irrelevant or insufficient evidence for biomedical questions.

### Dense Retrieval

Dense retrieval using `all-MiniLM-L6-v2` substantially outperformed the BM25 approaches.

With Top-3 retrieval:

- Accuracy: **0.740**
- Macro-F1: **0.660**

With Top-5 retrieval:

- Accuracy: **0.750**
- Macro-F1: **0.680**

Increasing Top-K from 3 to 5 produced a small improvement in both Accuracy and Macro-F1.

## Dataset

The project uses a BioASQ-derived biomedical question answering dataset containing binary `yes` / `no` questions.

The dataset is not included in this repository.

Expected local structure:

```text
data/
├── train.json
└── test.json
