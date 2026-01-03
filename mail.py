import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"F:\Desktop\taki_analysis\data\spam_ham_dataset.csv", encoding='latin-1', on_bad_lines='skip')
df = df.drop(columns=['Unnamed: 0','label_num'])
df = df.rename(columns={'label':'target','text':'text'})
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", '', x))
df['text'] = df['text'].apply(lambda x: re.sub(r"\S+@\S+", '', x))
df['text'] = df['text'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", '', x))
df = df.drop_duplicates(keep='first')
df = df.reset_index(drop=True)
df = df.dropna()

X = df['text']
y = df['target']
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

lengths = [len(x.split()) for x in X_train]
max_len = int(np.percentile(lengths,95))

X_train_pad = pad_sequences(X_train_seq,maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq,maxlen=max_len)

class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
class_weights_dict = dict(enumerate(class_weights))

model = Sequential()
model.add(Embedding(input_dim=10000,output_dim=128,input_length=max_len))
model.add(Bidirectional(LSTM(units=64,return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=32)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2)

history = model.fit(X_train_pad,y_train,epochs=10,batch_size=32,
                    validation_split=0.2,class_weight=class_weights_dict,callbacks=[early_stopping,lr_reducer])

model.save("spam_ham_model.h5")

y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test,y_pred,target_names=encoder.classes_))

def predict(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\S+@\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq,maxlen=max_len)
    pred_prob = model.predict(pad)
    pred = (pred_prob > 0.5).astype(int)
    label = encoder.inverse_transform(pred)[0]
    confidence = pred_prob[0][0] if label == 'spam' else 1 - pred_prob[0][0]
    return label , confidence

if __name__ == "__main__":
    test_text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
    label, confidence = predict(test_text)
    print(f"Text: {test_text}\nPredicted Label: {label}\nConfidence Score: {confidence:.2f}")
