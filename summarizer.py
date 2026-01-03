from transformers import pipeline 
summ = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text, max_length=130, min_length=30, do_sample=False):
    try:
        summary = summ(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error in summarization: {str(e)}"
if __name__ == "__main__":
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains every letter of the alphabet. "
        "It's often used to test fonts and keyboard layouts. "
        "In addition to its practical uses, it's also a fun example of a pangram."
    )
    summary = summarize_text(text)
    print("Original Text:\n", text)
    print("\nSummary:\n", summary)    