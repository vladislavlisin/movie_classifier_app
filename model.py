import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dropout, GRU, Embedding, Bidirectional, Dense, Flatten


class Model(object):

    # предпочтительнее работа с pandas и narray
    def __init__(self, input_length, idx2label, label2idx, embeddings=None, trainable=True, vocab_size=None):
        # макс. длина последовательности, массив эмбедингов, обучаемый ли эмбэдинг слой, количество меток, размерность словаря
        self.models = dict()
        self.idx2label = idx2label
        self.label2idx = label2idx
        self.threshold = 0.5
        self.labels_num = len(idx2label.keys())

        if embeddings is not None:
            self.embedding_dim = len(embeddings[0])  # размерность плотного векторного представления слова
            self.vocab_size = len(embeddings)  # количество слов в вокабуляре
        else:
            self.vocab_size = vocab_size
            self.embedding_dim = 100
        self.input_length = input_length

        self.gru_model = Sequential()
        self.gru_model.add(Embedding(input_dim=self.vocab_size,
                                     output_dim=self.embedding_dim,
                                     input_length=input_length,
                                     name='embed-layer',
                                     weights=[embeddings] if embeddings is not None else None,
                                     trainable=trainable
                                     ))
        self.gru_model.add(Bidirectional(GRU(256, return_sequences=True)))
        self.gru_model.add(Dropout(0.6))
        self.gru_model.add(Bidirectional(GRU(128, return_sequences=True)))
        self.gru_model.add(Dropout(0.5))
        self.gru_model.add(Bidirectional(GRU(64)))
        self.gru_model.add(Dropout(0.4))
        self.gru_model.add(Dense(32, activation='relu'))
        self.gru_model.add(Dense(16, activation='relu'))
        if self.labels_num > 2:
            self.gru_model.add(Dense(self.labels_num, activation='softmax'))

            self.gru_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss='categorical_crossentropy',
                metrics=["accuracy"]
            )
        else:
            self.gru_model.add(Dense(1, activation='sigmoid'))
            self.gru_model.compile(loss='binary_crossentropy',
                                   optimizer=tf.keras.optimizers.Adam(1e-3),
                                   metrics=['accuracy']
                                   )
        self.gru_model.summary()

    def checkpoint(self, model_name, model_save_path, monitor='val_loss'):
        self.models[model_name] = ModelCheckpoint(model_save_path,
                                                  monitor=monitor,
                                                  verbose=1,
                                                  save_best_only=True
                                                  )

    def train(self, X_train, y_train, model_name, epochs=20, batch_size=32):

        model_save_path = "./checks/" + model_name + '.h5'
        self.checkpoint(model_name, model_save_path, monitor='val_accuracy')

        history = self.gru_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[self.models[model_name]]
        )
        self.gru_model.load_weights(self.models[model_name].filepath)
        self.plot_graph(history)

    def predict(self, X_test, batch_size=None):
        # метод keras.model.predict возвращает именно вероятности а не логиты (всё окей!!!)
        y_prob = self.gru_model.predict(X_test, batch_size=batch_size)

        if self.labels_num == 2:
            preds = [1 if prob > self.threshold else 0 for prob in y_prob]
        else:
            preds = [np.argmax(x) for x in y_prob]

        mapped = list(map(lambda x: self.idx2label[x], preds))

        return mapped

    def evaluate(self, X_test, y_test, batch_size=32, threshold=0.5):

        self.threshold = threshold
        y_prob = self.gru_model.predict(X_test, batch_size=batch_size)

        if self.labels_num == 2:
            preds = [1 if prob > self.threshold else 0 for prob in y_prob]
        else:
            preds = [np.argmax(x) for x in y_prob]
            y_test = [np.argmax(x) for x in y_test]

        print("Confusion matrix: ")
        print(confusion_matrix(y_test, preds))

        print("Classification report: ")
        print(classification_report(y_test, preds))

        print("Accuracy score: ")
        print(accuracy_score(y_test, preds))

        return preds

    def plot_graph(self, history):
        plt.plot(history.history['accuracy'],
                 label='True answers part on TRAIN'
                 )
        plt.plot(history.history['val_accuracy'],
                 label='True answers part on VALID'
                 )
        plt.xlabel('train Epoch')
        plt.ylabel('True answers part')
        plt.legend()
        plt.show()

