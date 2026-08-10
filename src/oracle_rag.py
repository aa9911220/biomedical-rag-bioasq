import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================
# Configuration
# ==========================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TEST_PATH = "data/test.json"

OUTPUT_PATH = "results/oracle_predictions.json"

MAX_SAMPLES = None      # None = use all samples


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

with open(TEST_PATH, "r") as f:
    test_data = json.load(f)

if MAX_SAMPLES is not None:
    test_data = test_data[:MAX_SAMPLES]

print(f"Loaded {len(test_data)} samples.")


# ==========================
# Helper
# ==========================

def normalize_prediction(text):

    text = text.lower()

    if "yes" in text:
        return "yes"

    if "no" in text:
        return "no"

    return "unknown"


# ==========================
# Inference
# ==========================

predictions = []

for sample in tqdm(test_data):

    question = sample["question"]

    context = "\n".join(sample["context"])

    prompt = f"""You are a biomedical expert.

Use ONLY the evidence below to answer the question.

Evidence:
{context}

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
        "label": sample["label"]
    })


# ==========================
# Save
# ==========================

with open(OUTPUT_PATH, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\nSaved predictions to {OUTPUT_PATH}")
