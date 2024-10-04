from deep_translator import GoogleTranslator

def translate_text(text, source_language="auto", dest_language="en"):
    return GoogleTranslator(source=source_language, target=dest_language).translate(text)

def generate_tokens_from_caption(nlp, generated_caption):
    doc = nlp(generated_caption)
    objects = []
    for chunk in doc.noun_chunks:
        phrase = ' '.join(token.text for token in chunk if token.text.lower() not in ('a', 'an', 'the', 'picture', 'background'))
        if phrase.strip():
            objects.append(phrase.strip())
    return list(set(objects))
