from flask import request, jsonify
from PIL import Image
from modules.config_loaders import load_config
from modules.model_initializer import initialize_models
from modules.image_processing import extract_text, generate_caption
from modules.text_processing import translate_text, generate_tokens_from_caption, contains_chinese
from paddleocr import PaddleOCR
import logging
ppocr_logger = logging.getLogger('ppocr')
ppocr_logger.setLevel(logging.WARNING) 
# Load config and initialize models globally
config = load_config()
processor, model, nlp, device = initialize_models(config)

result_json = {
    config["result_json_keys"]["image_id_key"]: "",
    config["result_json_keys"]["image_caption"]: "caption will be here",
    config["result_json_keys"]["image_caption_tokens_key"]: "",
    config["result_json_keys"]["image_ocr_text_key"]: ""
}

def init_routes(app):
    @app.route('/upload', methods=['POST'])
    def upload():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file).convert('RGB')
            
            # Image captioning
            generated_text = generate_caption(processor, model, image, device)
            tokens_from_caption = generate_tokens_from_caption(nlp, generated_text)
            
            # OCR and Translation
            ocr = PaddleOCR(use_angle_cls=True, lang=config["ocr_language"])
            ocr_text = extract_text(ocr, image)

            print("OCR Text:", ocr_text)

            # Translate only the elements that contain Chinese characters
            translated_text = [
                translate_text(text)
                if contains_chinese(text) else text
                for text in ocr_text
            ]

            print("Translated Text:", translated_text)
            # translated_text = [translate_text(text, config["translator_src_language"], config["translator_des_language"]) for text in ocr_text]

            # Prepare response
            result_json[config["result_json_keys"]["image_id_key"]] = file.filename
            result_json[config["result_json_keys"]["image_caption"]] = generated_text
            result_json[config["result_json_keys"]["image_caption_tokens_key"]] = tokens_from_caption
            result_json[config["result_json_keys"]["image_ocr_text_key"]] = translated_text

            return result_json
        else:
            return jsonify({"error": "Invalid file type"}), 400
