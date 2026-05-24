import os
import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText

os.environ["HF_TOKEN"] = "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"

# 1. Load the lightweight SmolVLM model and processor
model_id = "HuggingFaceTB/SmolVLM-Instruct"

print("Loading model... (This might take a minute on the first run)")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Automatically uses your GPU
)

# 2. Get a test image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Formulate the prompt
user_prompt = "Describe the orientation of the car and its position relative to the background."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# 4. Process the inputs and run inference
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

print("\nRunning inference...")
generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

# 5. Output the result
print("\n--- VLM Output ---")
print(generated_texts[0].split("Assistant:")[-1].strip())
