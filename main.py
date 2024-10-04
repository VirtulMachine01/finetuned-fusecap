from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import torch
import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator
import yaml

app = Flask(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if(config["have_GPU"]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'


# Initialize models
processor = BlipProcessor.from_pretrained(config["huggingface_model_name"])
model = BlipForConditionalGeneration.from_pretrained(config["huggingface_model_name"]).to(device)
ocr = PaddleOCR(use_angle_cls=True, lang=config["ocr_language"])
nlp = spacy.load("en_core_web_sm")

def generate_tokens_from_caption(generated_caption):
    doc = nlp(generated_caption)
    objects = []
    for chunk in doc.noun_chunks:
        phrase = ' '.join(token.text for token in chunk if token.text.lower() not in ('a', 'an', 'the', 'picture', 'background'))
        if phrase.strip():
            objects.append(phrase.strip())
    return list(set(objects))

def extract_text(image):
    img_array = np.array(image)
    results = ocr.ocr(img_array, cls=True)
    if results[0] is None:
        return []
    texts_with_positions = [line[1][0] for result in results for line in result]
    return texts_with_positions

def translate_text(text, source_language=config["translator_src_language"], dest_language=config["translator_des_language"]):
    return GoogleTranslator(source=source_language, target=dest_language).translate(text)

def generate_caption(image):
    text = "a picture of "
    inputs = processor(image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs, num_beams=3)
    generated_text = processor.decode(out[0], skip_special_tokens=True)
    return generated_text


@app.route('/')
def index():
    return render_template('index.html')

result_json = {
    config["result_json_keys"]["image_id_key"]: "",
    config["result_json_keys"]["image_caption_tokens_key"]: "",
    config["result_json_keys"]["image_ocr_text_key"]: ""
}

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file).convert('RGB')
        
        generated_text = generate_caption(image)
        tokens_from_caption = generate_tokens_from_caption(generated_text)
        ocr_text = extract_text(image)
        translated_text = [translate_text(text) for text in ocr_text]
        
        result_json[config["result_json_keys"]["image_id_key"]] = file.filename
        result_json[config["result_json_keys"]["image_caption_tokens_key"]] = tokens_from_caption
        result_json[config["result_json_keys"]["image_ocr_text_key"]] = translated_text
        
        return result_json
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(debug=True)
