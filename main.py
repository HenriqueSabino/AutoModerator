from keras.saving import load_model
from fastapi import FastAPI
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from glob import glob
import nltk.corpus
from nltk.corpus import stopwords
import constants
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os

def load_tokenizer():
    if os.path.exists('./tokenizer/tokenizer.pickle'):
        with open('./tokenizer/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        df_nonflag = []
        df_flag =[]
        df_nonflag = pd.concat(
            [pd.read_parquet(x) for x in glob('./dataset/chats_nonflag_*.parquet')],
            ignore_index=True)
        df_flag = pd.concat(
            [pd.read_parquet(x) for x in glob('./dataset/chats_flagged_*.parquet')],
            ignore_index=True)

        df = pd.concat([df_nonflag, df_flag])
        df = df.drop_duplicates()

        df_labeled = df
        df_labeled["label"] = np.where(df["label"].str.contains("nonflagged"), 0, 1)
        df_labeled = df_labeled.sample(frac=1).reset_index(drop=True)

        nltk.download('stopwords')
        stop = stopwords.words('english')

        Words = []
        Labels = []
        for item in df_labeled.body :
            item = " ".join([word for word in item.split() if word not in (stop)])
            Words.append(item)

        for item in df_labeled.label :
            Labels.append(item)

        rus = RandomUnderSampler()
        Words = np.array(Words)
        Words_reshape = Words.reshape(-1, 1)
        X_rus, y_rus= rus.fit_resample(Words_reshape, Labels)

        training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(X_rus, y_rus, test_size=0.15, random_state=42)

        training_sentences = training_sentences.tolist()

        tokenizer = Tokenizer(num_words=constants.VOCAB_SIZE, oov_token=constants.OOV_TOK)
        tokenizer.fit_on_texts(training_sentences)

        with open('./tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer

tokenizer = load_tokenizer()

app = FastAPI()

@app.post("/should_flag_message", )
def should_flag_message(message: str) -> bool:

    automod_model = load_model('./model/model.keras', compile=True, safe_mode=True)

    vectorized_message = tokenizer.texts_to_sequences([message])
    padded_message = pad_sequences(vectorized_message, maxlen=constants.MAX_LENGTH, padding=constants.PADDING_TYPE, truncating=constants.TRUNC_TYPE)
    
    should_flag = automod_model.predict(padded_message)[0][0] > 0.5

    return str(should_flag)