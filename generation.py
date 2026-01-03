import torch
from transformers import RagTokenizer, RagSequenceForGeneration

MODEL_NAME = "facebook/rag-token-nq"

class EnhancedRetriever:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = RagTokenizer.from_pretrained(MODEL_NAME)
        self.model = RagSequenceForGeneration.from_pretrained(MODEL_NAME)
        self.model.to(self.device)

    def generate(self, query, max_length=50):
        inputs = self.tokenizer(
            query,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length
            )

        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

    def main(self):
        query = "What is the capital of France?"
        result = self.generate(query)
        print("Answer:", result)

if __name__ == "__main__":
    retriever = EnhancedRetriever()
    retriever.main()
