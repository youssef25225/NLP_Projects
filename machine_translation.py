from transformers import pipeline

model = "Helsinki-NLP/opus-mt-en-ar"
pipeline = pipeline("translation", model=model)

def translate_text(text: str) -> str:
    if not text.strip():
        return "Please enter a valid text"
    try:
        result = pipeline(text, max_length=256, truncation=True)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error in translation: {str(e)}"

if __name__ == "__main__":
    sample = "I have 3 books! And you?"
    print(translate_text(sample))
