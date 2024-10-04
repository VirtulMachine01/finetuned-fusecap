from flask import request, jsonify, render_template
from PIL import Image
from modules.config_loaders import load_config
from modules.model_initializer import initialize_models
from modules.image_processing import extract_text, generate_caption
from modules.text_processing import translate_text, generate_tokens_from_caption

# Load config and initialize models globally
config = load_config()
processor, model, ocr, nlp, device = initialize_models(config)

result_json = {
    config["result_json_keys"]["image_id_key"]: "",
    config["result_json_keys"]["image_caption_tokens_key"]: "",
    config["result_json_keys"]["image_ocr_text_key"]: ""
}

def init_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

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
            ocr_text = extract_text(ocr, image)
            translated_text = [translate_text(text, config["translator_src_language"], config["translator_des_language"]) for text in ocr_text]
            
            # Prepare response
            result_json[config["result_json_keys"]["image_id_key"]] = file.filename
            result_json[config["result_json_keys"]["image_caption_tokens_key"]] = tokens_from_caption
            result_json[config["result_json_keys"]["image_ocr_text_key"]] = translated_text

            return result_json
        else:
            return jsonify({"error": "Invalid file type"}), 400
