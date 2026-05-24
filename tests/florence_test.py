import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "microsoft/Florence-2-base"
print("Loading Florence-2 on stable Transformers...")

# 1. Clean loading with remote code trusted and eager attention
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    attn_implementation="eager"
).to("cuda")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 2. Grab a test image
url = "https://images.unsplash.com/photo-1531746790731-6c087fecd65a?q=80&w=1200"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Use Florence-2's Object Detection task token
task_prompt = "<OD>"

# 4. Format inputs for the GPU
inputs = processor(text=task_prompt, images=image, return_tensors="pt")
inputs["input_ids"] = inputs["input_ids"].to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to("cuda", torch.float16)

print("\nRunning Object Detection Inference...")
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3
)

# 5. Decode and post-process the results into text/coordinates
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(
    generated_text, 
    task=task_prompt, 
    image_size=(image.width, image.height)
)

print("\n--- Florence-2 Coordinates Output ---")
print(parsed_answer)
