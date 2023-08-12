import os.path
import pickle
import pandas as pd
import numpy as np
import os.path

from gensim.models import KeyedVectors
import gensim.downloader as api

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow.python.client import device_lib

#from model import Model
#from data import Data

###############################################################################
#                                ALGO PART                                    #
###############################################################################


def get_w2v_model():
    print("Start getting embs")
    fname = "./data/vectors"

    if not os.path.isfile(fname):
        wv = api.load('word2vec-google-news-300')
        wv.save(fname)
    else:
        wv = KeyedVectors.load(fname, mmap='r')

    print("End getting embs")
    return wv


# преобразуем последовательности чисел в плотные векторные представления

def get_w2v_embeddings(train_word_index, model, EMBEDDING_DIM=300):
    train_embedding_weights = np.zeros((len(train_word_index), EMBEDDING_DIM))
    unknown = 0

    for word, idx in train_word_index.items():
        if word in model:
            train_embedding_weights[idx, :] = model[word]
        else:
            unknown += 1
            np.random.rand(EMBEDDING_DIM)
    print(unknown)
    return train_embedding_weights


def pipeline(X, model_name, data_title, w2v_model):
    print("Start data fit")

    data_path = "./data/" + data_title
    if os.path.isfile(data_path):
      with open(data_path, "rb") as fp:  # Unpickling
        X = pickle.load(fp)
    else:
      X.fit()
      X.preprocess()
      with open(data_path, "wb") as fp:  # Pickling
        pickle.dump(X, fp)

    X_train, X_test, y_train, y_test = train_test_split(X.pad_cut_seq,
                                                        X.y,
                                                        train_size=0.80,
                                                        test_size=0.20,
                                                        random_state=1,
                                                        stratify=X.y)

    print("End data fit")
    print("Start model fit")

    input_length = X_train.shape[1]

    # если модель есть - подгружаем, иначе берём эмбединги, создаём модель, обучаем и дэмпим
    model_path = "./models/" + model_name
    if os.path.isfile(model_path):
        with open(model_path, "rb") as fp:  # Unpickling
          gru = pickle.load(fp)
    else:
        print("Start emb fit")
        embeddings_path = "./data/w2v_" + data_title
        if os.path.isfile(embeddings_path):
            with open(embeddings_path, "rb") as fp:  # Unpickling
                en_w2v_emb = pickle.load(fp)
        else:
            en_w2v_emb = get_w2v_embeddings(X.tokenizer.word_index, w2v_model, EMBEDDING_DIM=300)
            with open(embeddings_path, "wb") as fp:  # Pickling
                pickle.dump(en_w2v_emb, fp)
        print("End emb fit")
        gru = Model(input_length, X.idx2label, X.label2idx, embeddings=en_w2v_emb,
                    trainable=False, vocab_size=None)
        gru.train(X_train, y_train, model_name=model_name, epochs=10, batch_size=32)
        _ = gru.evaluate(X_test, y_test, batch_size=32, threshold=0.5)
        with open(model_path, "wb") as fp:  # Pickling
            pickle.dump(gru, fp)

    print("End model fit")

    return gru, X


def input():
    print("Start data input")

    df = pd.read_csv("./data/quarter_dataset")

    # prepare data for sentiment model

    y_sentiment = df.Sentiment
    review = df.Review

    idx2label_senti = {
        0: "Negative",
        1: "Positive",
        2: "Neutral",
    }
    label2idx_senti = {
        "Negative": 0,
        "Positive": 1,
        "Neutral": 2
    }

    temp = to_categorical(y_sentiment, num_classes=3)
    x = Data(review, temp, label2idx_senti, idx2label_senti)

    # prepare data for rating models

    data_1_4 = df.loc[df["Rating"] < 5]
    data_5_6 = df.loc[df.Rating.isin([5, 6])]
    data_7_10 = df.loc[df.Rating > 6]

    label2idx_7_10 = {7: 0, 8: 1, 9: 2, 10: 3}
    idx2label_7_10 = {0: 7, 1: 8, 2: 9, 3: 10}

    label2idx_5_6 = {5: 0, 6: 1}
    idx2label_5_6 = {0: 5, 1: 6}

    label2idx_1_4 = {1: 0, 2: 1, 3: 2, 4: 3}
    idx2label_1_4 = {0: 1, 1: 2, 2: 3, 3: 4}

    temp_5_6 = data_5_6.Rating.map(label2idx_5_6)

    temp_7_10 = data_7_10.Rating.map(label2idx_7_10)
    temp_7_10 = to_categorical(temp_7_10, num_classes=4)

    temp_1_4 = data_1_4.Rating.map(label2idx_1_4)
    temp_1_4 = to_categorical(temp_1_4, num_classes=4)

    # Positive - 1
    x_7_10 = Data(data_7_10.Review, temp_7_10, label2idx_7_10, idx2label_7_10)
    # Neutral - 2
    x_5_6 = Data(data_5_6.Review, temp_5_6, label2idx_5_6, idx2label_5_6)
    # Negative - 0
    x_1_4 = Data(data_1_4.Review, temp_1_4, label2idx_1_4, idx2label_1_4)

    data = [x_7_10, x, x_1_4, x_5_6]

    print("End data input")

    return data



def action():

    models = {}
    datas = {}

    # загружаем модели и дату, если они есть
    model_names = ["gru_1", "gru", "gru_0","gru_2"]
    data_names = ["data_1", "data_senti", "data_0", "data_2"]

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())

    print("Start action")

    data = input()

    w2v_model = get_w2v_model()

    # обучаем 4 модели
    for i, j, k in zip(data, model_names, data_names):

        print(j)
        print(k)
        print(i.shape())
        models[j], datas[k] = pipeline(i, j, k, w2v_model)

    print("End action")

    return models, datas


if __name__ == "__main__":
    action()