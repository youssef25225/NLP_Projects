from langdetect import detect
import nltk
import re

nltk.download("punkt", quiet=True)


def analyze_text(text: str) -> str:
    if not text.strip():
        return "الرجاء إدخال نص للتشخيص"

    diagnostics = []

    try:
        lang = detect(text)
        diagnostics.append(f" اللغة المكتشفة: {lang}")
    except:
        diagnostics.append(" لم أتمكن من تحديد اللغة")

    diagnostics.append(f" عدد الكلمات: {len(text.split())}")
    diagnostics.append(f" عدد الحروف: {len(text)}")

    if re.search(r'\d', text):
        diagnostics.append(" يحتوي النص على أرقام")

    punct = re.findall(r'[,.!?؟]', text)
    diagnostics.append(f" علامات الترقيم: {len(punct)}")

    return "\n".join(diagnostics)


if __name__ == "__main__":
    sample = "انا عندي 3 كتب! وانت؟"
    print(analyze_text(sample))
