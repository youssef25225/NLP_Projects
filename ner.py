import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
class NER:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").to(self.device)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def preprocessing_text(self, text):
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    def main(self):
        text = input("Enter your text: ")
        if not text or not text.strip():
            print("Please enter a valid text")
            return
        try:
            result = self.nlp(text)
            print(result)
        except Exception as e:
            print("Error in NER: ", e)

if __name__ == "__main__":
    ner = NER()
    ner.main()
