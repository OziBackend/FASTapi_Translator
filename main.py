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
        model_dir = f"./models/{src_lang}_{tgt_lang}"  # Directory to save/load models

        if model_name not in self.models:  # Load only if not already loaded
            try:
                # Try loading the model and tokenizer from the saved directory
                self.tokenizers[model_name] = MarianTokenizer.from_pretrained(model_dir)
                self.models[model_name] = MarianMTModel.from_pretrained(model_dir).to(self.device)
                print("Loaded model and tokenizer from disk.")
            except Exception as e:
                # If loading fails, download and save them
                print("Loading from disk failed, downloading new model.")
                self.tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
                self.models[model_name] = MarianMTModel.from_pretrained(model_name).to(self.device)
                self.tokenizers[model_name].save_pretrained(model_dir)
                self.models[model_name].save_pretrained(model_dir)
                print("Model and tokenizer saved to disk.")

        return self.models[model_name], self.tokenizers[model_name]

    def translate(self, text, src_lang, tgt_lang):
        model, tokenizer = self.load_model(src_lang, tgt_lang)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        translated = model.generate(**inputs)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

translator = Translator()

@app.on_event("startup")
def load_default_model():
    translator.load_model("en", "fr")  # Preload a default model at startup

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate/")
def translate_text(request: TranslationRequest):
    translation = translator.translate(request.text, request.source_lang, request.target_lang)
    return {"translated_text": translation}
