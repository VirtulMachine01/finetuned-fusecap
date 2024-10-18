from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
local_model_path = "models/mbart_50"

model = MBartForConditionalGeneration.from_pretrained(local_model_path)
tokenizer = MBart50TokenizerFast.from_pretrained(local_model_path)

def translate_text(text):
    print("Translating Chunk : ", text)
    tokenizer.src_lang = "zh_CN"
    encoded_ch = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_ch)
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text[0]


# from deep_translator import GoogleTranslator

# def translate_text(text, source_language="auto", dest_language="en"):
#     print("Translating chunk : ", text)
#     return GoogleTranslator(source=source_language, target=dest_language).translate(text)

def generate_tokens_from_caption(nlp, generated_caption):
    doc = nlp(generated_caption)
    objects = []
    for chunk in doc.noun_chunks:
        phrase = ' '.join(token.text for token in chunk if token.text.lower() not in ('a', 'an', 'the', 'picture', 'background'))
        if phrase.strip():
            objects.append(phrase.strip())
    return list(set(objects))

import re
def contains_chinese(text):
    """Check if the given text contains any Chinese characters."""
    # Regex pattern to match Chinese characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))