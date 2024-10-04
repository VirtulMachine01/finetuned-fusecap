import numpy as np
from PIL import Image

def extract_text(ocr, image):
    img_array = np.array(image)
    results = ocr.ocr(img_array, cls=True)
    if results[0] is None:
        return []
    texts_with_positions = [line[1][0] for result in results for line in result]
    return texts_with_positions

def generate_caption(processor, model, image, device):
    text = "a picture of "
    inputs = processor(image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs, num_beams=3)
    generated_text = processor.decode(out[0], skip_special_tokens=True)
    return generated_text
