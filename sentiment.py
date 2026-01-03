from transformers import pipeline
arabic = pipeline('text-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment')
english = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
def analyze_sentiment(text):
    try:
        if any('\u0600' <= char <= '\u06FF' for char in text):
            result = arabic(text)[0]
        else:
            result = english(text)[0]
        label = result['label']
        confidence = result['score']
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"
if __name__ == "__main__":
    test_text_en = "I love programming! It's so much fun."
    test_text_ar = "أنا أحب البرمجة! إنه ممتع جدا."
    result_en = analyze_sentiment(test_text_en)
    result_ar = analyze_sentiment(test_text_ar)
    print(f"English Text: {test_text_en}\nSentiment Analysis: {result_en}\n")
    print(f"Arabic Text: {test_text_ar}\nSentiment Analysis: {result_ar}\n")    