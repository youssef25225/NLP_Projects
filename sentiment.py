import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
class SentimentAnalysis:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.model = AutoModelForTokenClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1").to(self.device)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def main(self):
        text = input("Enter your text: ")
        if not text or not text.strip():
            print("Please enter a valid text")
            return
        try:
            result = self.nlp(text)
            print(result)
        except Exception as e:
            print("Error: ", e)

if __name__ == "__main__":
    sentiment = SentimentAnalysis()
    sentiment.main()

