import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi

# ==========================
# Configuration
# ==========================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_PATH = "data/train.json"
TEST_PATH = "data/test.json"

OUTPUT_PATH = "results/bm25_doc_predictions.json"

TOP_K = 2
MAX_SAMPLES = None

# ==========================
# Load model
# ==========================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# ==========================
# Load data
# ==========================

with open(TRAIN_PATH, "r") as f:
    train_data = json.load(f)

with open(TEST_PATH, "r") as f:
    test_data = json.load(f)

if MAX_SAMPLES is not None:
    test_data = test_data[:MAX_SAMPLES]

print(f"Train samples: {len(train_data)}")
print(f"Test samples : {len(test_data)}")

# ==========================
# Build BM25 corpus
# ==========================

corpus = []

for sample in train_data:
    document = " ".join(sample["context"])

    corpus.append(document)

print(f"Corpus size: {len(corpus)}")

tokenized_corpus = [
    doc.lower().split()
    for doc in corpus
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index built.")

# ==========================
# Helper
# ==========================

def normalize_prediction(text):

    text = text.lower().strip()

    if text.startswith("yes"):
        return "yes"

    if text.startswith("no"):
        return "no"

    if " yes" in text:
        return "yes"

    if " no" in text:
        return "no"

    return "unknown"

# ==========================
# Inference
# ==========================

predictions = []

for sample in tqdm(test_data):

    question = sample["question"]

    tokenized_query = question.lower().split()

    retrieved_docs = bm25.get_top_n(
        tokenized_query,
        corpus,
        n=TOP_K
    )

    evidence = "\n".join(retrieved_docs)

    prompt = f"""You are a biomedical expert.

Use ONLY the retrieved evidence below.

Evidence:
{evidence}

Question:
{question}

Answer ONLY one word.

yes

or

no
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    prediction = normalize_prediction(generated)

    predictions.append({
        "id": sample["id"],
        "question": question,
        "prediction": prediction,
        "label": sample["label"],
        "retrieved_context": retrieved_docs
    })

# ==========================
# Save
# ==========================

with open(OUTPUT_PATH, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\nSaved predictions to {OUTPUT_PATH}")
