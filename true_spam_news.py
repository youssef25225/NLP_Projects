import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")



def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()


df_true = pd.read_csv(
    r'F:\Desktop\taki_analysis\data\True.csv', on_bad_lines='skip')
df_fake = pd.read_csv(
    r'F:\Desktop\taki_analysis\data\Fake.csv', on_bad_lines='skip')

df_true['label'] = 1
df_fake['label'] = 0

df = pd.concat([df_true, df_fake], ignore_index=True)
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

vec = TfidfVectorizer(stop_words='english', max_df=0.7,
                      ngram_range=(1, 2), max_features=5000)
model = LogisticRegression(class_weight="balanced", max_iter=1000)

pipeline = make_pipeline(vec, model)
pipeline.fit(X_train, y_train)



def news(text):
    text = clean_text(text)
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    confidence = np.max(probabilities)
    label = "True News" if prediction == 1 else "Fake News"
    return {"label": label, "confidence": float(confidence)}


if __name__ == "__main__":
    text = "The economy is improving and jobs are being created."
    result = news(text)
    print(result)

    y_pred = pipeline.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
