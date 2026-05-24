import os
from pathlib import Path
import urllib.request
from PIL import Image, ImageDraw, ImageFont

def create_scenes():
    # Create directory
    base_dir = Path("test_images")
    base_dir.mkdir(exist_ok=True)
    print(f"📁 Created folder: {base_dir.absolute()}")

    # Move existing sample scene if it's still in the root folder
    old_scene = Path("sample_scene.png")
    if old_scene.exists():
        old_scene.rename(base_dir / "sample_scene.png")
        print("📦 Moved sample_scene.png into test_images/")

    # --- Generate OCR Test Scene ---
    print("🎨 Generating OCR test scene...")
    img = Image.new('RGB', (1024, 1024), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # Draw a mock "box"
    draw.rectangle([200, 300, 800, 700], fill='#c2a370', outline='#8b7355', width=5)
    # Draw a "label"
    draw.rectangle([300, 450, 700, 550], fill='white')
    
    # Try to load a default font, otherwise use whatever PIL has
    try:
        font = ImageFont.truetype("DejavuSans-Bold.ttf", 80)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((320, 460), "PARTS: SCREWS", fill='black', font=font)
    draw.text((320, 510), "QTY: 500", fill='red', font=font)
    
    ocr_path = base_dir / "ocr_test.png"
    img.save(ocr_path)
    print(f"✅ Saved OCR test image: {ocr_path}")

    # --- Download a complex real-world scene ---
    print("🌍 Downloading a complex robot table scene...")
    complex_path = base_dir / "complex_scene.jpg"
    try:
        url = "https://images.unsplash.com/photo-1581092160562-40aa08e78837?q=80&w=1024&auto=format&fit=crop"
        urllib.request.urlretrieve(url, complex_path)
        print(f"✅ Saved complex test image: {complex_path}")
    except Exception as e:
        print(f"⚠ Could not download image: {e}")

if __name__ == "__main__":
    create_scenes()
    print("\n🎉 Done! You can now test the new images.")
