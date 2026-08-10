Experiment 1:
Model: Qwen2.5-1.5B-Instruct
Setting: Question only
Samples: 292

Accuracy: 0.6267
Macro-F1: 0.5794


Experiment 2:
Model: Qwen2.5-1.5B-Instruct
Setting: Oracle context
Samples: 292

Accuracy: 0.8870
Macro-F1: 0.8486

Experiment 3a:
BM25 Sentence-level RAG (lexical retrieval)
Accuracy:0.37
Macro-F1:0.35

Experiment 3b:
BM25 Document-level RAG (lexical retrieval)
Accuracy:0.42
Macro-F1:0.41

Experiment 4a:
Dense RAG(semantic retrieval (MiniLM)
Accuracy:0.74
Macro-F1:0.66

Experiment 4b:
Dense RAG(semantic retrieval (MiniLM)
Accuracy:0.75
Macro-F1:0.68
