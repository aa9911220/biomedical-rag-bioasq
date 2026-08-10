import json
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ======================
# Paths
# ======================

TRAIN_PATH = "data/train.json"
TEST_PATH = "data/test.json"

OUTPUT_PATH = "results/dense_top5_predictions.json"


# ======================
# Models
# ======================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


TOP_K = 5


# ======================
# Device
# ======================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ======================
# Load embedding model
# ======================

print("Loading embedding model...")

embedder = SentenceTransformer(
    EMBED_MODEL,
    device=device
)


# ======================
# Load data
# ======================

with open(TRAIN_PATH, "r") as f:
    train_data = json.load(f)

with open(TEST_PATH, "r") as f:
    test_data = json.load(f)


print("Train samples:", len(train_data))
print("Test samples :", len(test_data))


# ======================
# Build sentence corpus
# ======================

corpus = []

for sample in train_data:
    for sentence in sample["context"]:
        sentence = sentence.strip()

        if sentence:
            corpus.append(sentence)


print("Corpus size:", len(corpus))


# ======================
# Encode corpus
# ======================

print("Encoding corpus...")

corpus_embeddings = embedder.encode(
    corpus,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True
)


print("Embedding finished.")


# ======================
# Load LLM
# ======================

print("Loading LLM...")


tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)


model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)


# ======================
# Retrieval
# ======================

def retrieve(question):

    query_embedding = embedder.encode(
        question,
        convert_to_tensor=True
    )


    scores = cosine_similarity(
        query_embedding.cpu().numpy().reshape(1,-1),
        corpus_embeddings.cpu().numpy()
    )[0]


    top_indices = scores.argsort()[-TOP_K:][::-1]


    return [
        corpus[i]
        for i in top_indices
    ]



# ======================
# Inference
# ======================

results = []


for item in tqdm(test_data):

    question = item["question"]


    retrieved_context = retrieve(
        question
    )


    context_text = "\n".join(
        [
            f"Evidence {i+1}: {c}"
            for i,c in enumerate(retrieved_context)
        ]
    )


    prompt = f"""
You are a biomedical question answering system.

Answer the question with only one word:
yes or no.

Question:
{question}

Evidence:
{context_text}

Answer:
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False
    )


    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    answer = generated.split("Answer:")[-1].strip().lower()


    if "yes" in answer:
        prediction = "yes"
    elif "no" in answer:
        prediction = "no"
    else:
        prediction = "no"



    results.append(
        {
            "id": item["id"],
            "question": question,
            "prediction": prediction,
            "label": item["label"],
            "retrieved_context": retrieved_context
        }
    )


# ======================
# Save
# ======================

with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print(
    f"Saved predictions to {OUTPUT_PATH}"
)
