from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch

app = FastAPI()

class Translator:
    def __init__(self):
        self.models = {}  # Dictionary to hold models for different language pairs
        self.tokenizers = {}  # Dictionary to hold tokenizers for different language pairs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, src_lang, tgt_lang):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        if model_name not in self.models:  # Load only if not already loaded
            self.tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
            self.models[model_name] = MarianMTModel.from_pretrained(model_name).to(self.device)
        return self.models[model_name], self.tokenizers[model_name]

    def translate(self, text, src_lang, tgt_lang):
        model, tokenizer = self.load_model(src_lang, tgt_lang)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        translated = model.generate(**inputs)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

translator = Translator()

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate/")
def translate_text(request: TranslationRequest):
    translation = translator.translate(request.text, request.source_lang, request.target_lang)
    return {"translated_text": translation}
