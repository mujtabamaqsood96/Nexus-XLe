import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def main():
    print("Loading model...")
    model_id = "microsoft/Florence-2-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)

    image_path = "/home/twinxblaze/fyp/test_images/ocr_test.jpg"
    print(f"Loading image {image_path}...")
    image = Image.open(image_path).convert("RGB")

    task = "<OCR_WITH_REGION>"
    print(f"Running task: {task}...")
    
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"Raw generated text:\n{generated_text}")
    
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height)
    )
    print(f"\nParsed answer:\n{parsed_answer}")

if __name__ == "__main__":
    main()
