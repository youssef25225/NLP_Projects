import re, Counter, nltk
import numpy as np

def analyze_text(text: str) -> str:
    if not text.strip():
        return "Please enter a valid text"
    else:
        diagnostics = []
        try:
            lang = detect(text)
            diagnostics.append(f"Detected language: {lang}")
        except:
            diagnostics.append("Unable to detect language")
        diagnostics.append(f" Number of words: {len(text.split())}")
        diagnostics.append(f" Number of characters: {len(text)}")
        if re.search(r'\d', text):
            diagnostics.append(" The text contains numbers")
        punct = re.findall(r'[,.!?؟]', text)
        diagnostics.append(f" Number of punctuation marks: {len(punct)}")
        return "\n".join(diagnostics)
def main():
    text = input("Enter your text: ")
    result = analyze_text(text)
    print(result)
if __name__ == "__main__":
    main()
