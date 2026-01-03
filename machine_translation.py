from transformers import pipeline

modelName = "Helsinki-NLP/opus-mt-en-ar"

translator = pipeline(
    task="translation",
    model=modelName
)

def translate_text(text: str) -> str:
    if not text or not text.strip():
        return "Please enter a valid text"

    try:
        result = translator(text, max_length=256, num_beams=5, do_sample=True)
        return result[0]["translation_text"]
    except Exception as e:
        return f"Translation error: {e}"

if __name__ == "__main__":
    sample = input("Enter your text: ").strip().lower()
    if not sample:
        print("Please enter a valid text")
        return
    try:
        result = translate_text(sample)
        print("Text:\n", sample)
        print("\nTranslated Text:\n", result)
    except Exception as e:
        print("Error: ", e)

