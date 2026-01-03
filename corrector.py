import re
import torch
from langdetect import detect, LangDetectException
from transformers import pipeline

MODEL_ENGLISH = "hassaanik/grammar-correction-model"
MODEL_ARABIC = "qcri/arabic-error-correction"

device = 0 if torch.cuda.is_available() else -1

try:
    grammar_pipeline_ar = pipeline("text2text-generation", model=MODEL_ARABIC, device=device)
except Exception as e:
    print("Arabic model load failed:", e)
    grammar_pipeline_ar = None

try:
    grammar_pipeline_en = pipeline("text2text-generation", model=MODEL_ENGLISH, device=device)
except Exception as e:
    print("English model load failed:", e)
    grammar_pipeline_en = None


def correct_text(text: str) -> str:
    if not text.strip():
        return "الرجاء إدخال نص صالح"

    try:
        lang = detect(text)
    except LangDetectException:
        lang = "en"

    lang = "ar" if lang.startswith("ar") else "en"
    pipeline_model = grammar_pipeline_ar if lang == "ar" else grammar_pipeline_en

    if pipeline_model is None:
        return f"خدمة التصحيح غير متوفرة للغة {lang}"

    try:
        result = pipeline_model(text, max_length=256, truncation=True)
        return result[0]["generated_text"]
    except Exception as e:
        return f"خطأ في التصحيح: {str(e)}"


if __name__ == "__main__":
    print(correct_text("انا بحب البرمجه جدا و اريد ان اتعلم."))
    print(correct_text("He dont know where he goed yesterday."))
