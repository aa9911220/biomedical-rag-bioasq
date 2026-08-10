import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================
# Configuration
# =====================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = "data/test.json"
OUTPUT_PATH = "results/baseline_predictions.json"


MAX_SAMPLES = None


# =====================
# Load dataset
# =====================

with open(DATA_PATH, "r") as f:
    data = json.load(f)

if MAX_SAMPLES:
    data = data[:MAX_SAMPLES]

print(f"Loaded {len(data)} samples")


# =====================
# Load model
# =====================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("Model loaded")


# =====================
# Inference
# =====================

results = []


for item in tqdm(data):

    question = item["question"]

    prompt = f"""
You are a medical expert.

Answer the following question with only one word:
yes, no, or maybe.

Question:
{question}

Answer:
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )


    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )


    prediction = generated.strip().lower()


    # clean output
    if "yes" in prediction:
        prediction = "yes"
    elif "no" in prediction:
        prediction = "no"
    elif "maybe" in prediction:
        prediction = "maybe"
    else:
        prediction = "unknown"


    results.append(
        {
            "id": item["id"],
            "question": question,
            "prediction": prediction,
            "label": item["label"]
        }
    )


# =====================
# Save
# =====================

with open(OUTPUT_PATH, "w") as f:
    json.dump(
        results,
        f,
        indent=2
    )


print(f"Saved predictions to {OUTPUT_PATH}")

