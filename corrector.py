from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import torch
class Corrector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("hassaanik/grammar-correction-model")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("hassaanik/grammar-correction-model").to(self.device)
        self.nlp = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
    def correct_grammar(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    def main(self):
        text = input("Enter your text: ").strip().lower()
        if not text:
            print("Please enter a vald text")
        try:
            corrected_text = self.correct_grammar(text)
            print("Original: ", text)
            print("\nCorrecte: ", corrected_text)
        except Exception as e:
            print("Error", e)
if __name__ == "__main__":
    corrector = Corrector()
    corrector.main()

