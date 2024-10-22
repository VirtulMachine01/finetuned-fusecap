from flask import request, jsonify
from PIL import Image
from modules.config_loaders import load_config
from modules.model_initializer import initialize_models
from modules.image_processing import extract_text, generate_caption
from modules.text_processing import translate_text, generate_tokens_from_caption, contains_chinese
from paddleocr import PaddleOCR
import magic
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

def is_image(file):
    # Create a Magic object to check the MIME type
    mime = magic.Magic(mime=True)
    # Get the MIME type from the file's bytes
    file_type = mime.from_buffer(file.read(1024))
    # Reset file pointer to the beginning after reading
    file.seek(0)
    # Check if the MIME type starts with 'image/'
    return file_type.startswith('image/')

def contains_novideo(lst):
    # Normalize the variations of "novideo" by removing spaces and converting to lowercase
    normalized_list = [''.join(item.lower().split()) for item in lst]
    # Check if any normalized item matches "novideo"
    return 'novideo' not in normalized_list

def init_routes(app):
    @app.route('/upload', methods=['POST'])
    def upload():
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and is_image(file):
        # if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file).convert('RGB')
            
            # OCR and Translation
            ocr = PaddleOCR(use_angle_cls=True, lang=config["ocr_language"])
            ocr_text = extract_text(ocr, image)

            if contains_novideo(ocr_text):
                # Image captioning
                print("Video is There")
                generated_text = generate_caption(processor, model, image, device)
                tokens_from_caption = generate_tokens_from_caption(nlp, generated_text)
            else:
                generated_text = "Video is not available"
                tokens_from_caption = ["No Video", "Null"]
            
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
            return jsonify({f"{config["result_json_keys"]["image_id_key"]}":file.filename,"error": "Invalid file type"}), 400
