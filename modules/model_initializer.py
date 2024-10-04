import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR
import spacy

def initialize_models(config):
    # Device configuration
    device = torch.device('cuda' if config["have_GPU"] and torch.cuda.is_available() else 'cpu')
    
    # Initialize BLIP model for image captioning
    processor = BlipProcessor.from_pretrained(config["huggingface_model_name"])
    model = BlipForConditionalGeneration.from_pretrained(config["huggingface_model_name"]).to(device)
    
    # Initialize PaddleOCR for text extraction
    ocr = PaddleOCR(use_angle_cls=True, lang=config["ocr_language"])
    
    # Load SpaCy for text processing
    nlp = spacy.load("en_core_web_sm")
    
    return processor, model, ocr, nlp, device
