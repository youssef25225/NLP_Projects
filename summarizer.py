from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 

class Summarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(self.device)
        self.nlp = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

    def main(self):
        text = input("Enter your text: ")
        if not text or not text.strip():
            print("Please enter a valid text")
            return
        try:
            result = self.nlp(text)
            print(result)
        except Exception as e:
            print("Error in summarization: ", e)

if __name__ == "__main__":
    summarizer = Summarizer()
    summarizer.main()
