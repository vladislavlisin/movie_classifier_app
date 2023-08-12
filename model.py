from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dropout, Activation, GRU, Embedding, Bidirectional, Dense, Flatten

class model():

# предпочтительнее работа с pandas и narray
    def __init__(self, input_length, embeddings=None, trainable=True, labels_num=2, vocab_size=None):
        # макс. длина последовательности, массив эмбедингов, обучаемый ли эмбэдинг слой, количество меток, размерность словаря
        self.models = dict()

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
        self.gru_model.add(GRU(256, return_sequences=True))
        self.gru_model.add(Dropout(0.6))
        self.gru_model.add(GRU(128, return_sequences=True))
        self.gru_model.add(Dropout(0.5))
        self.gru_model.add(GRU(64, return_sequences=True))
        self.gru_model.add(Dropout(0.4))
        self.gru_model.add(Dense(32, activation='relu'))
        if labels_num > 2:
            self.gru_model.add(Flatten())
            self.gru_model.add(Dense(labels_num, activation='softmax'))

            self.gru_model.compile(
                optimizer=Adam(1e-3),
                loss='categorical_crossentropy',
                metrics=["accuracy"]
            )
        else:
            self.gru_model.add(Dense(1, activation='sigmoid'))
            self.gru_model.compile(loss='binary_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy']
                              )
        self.gru_model.summary()

    def train(self, X_train, y_train, model_name, epochs=20, batch_size=32):

        self.checkpoint(self.models, model_name, monitor='val_accuracy')

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


    def predict(self, X_test, batch_size=32):
        # метод keras.model.predict возвращает именно вероятности а не логиты (всё окей!!!)
        y_prob = self.gru_model.predict(X_test, batch_size=batch_size)

        return y_prob

    def evaluate(self, X_test, y_test, batch_size=32, threshold=0.5):

        y_prob = self.predict(X_test, batch_size=batch_size)

        preds = [1 if prob > threshold else 0 for prob in y_prob]

        real = Counter(y_test)
        print('real: ', real)
        cnt = Counter(preds)
        print("preds: ", cnt)

        print("Confusion matrix: ")
        print(confusion_matrix(y_test, preds))

        print("Classification report: ")
        print(classification_report(y_test, preds))

        print("Here: ", {"ROC AUC": roc_auc_score(y_test, y_prob),
                         "Balanced Accuracy": balanced_accuracy_score(y_test, preds),
                         "F1": f1_score(y_test, preds)})

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

    def checkpoint(self, model_name, model_save_path_model_name, monitor='val_loss'):
        model_save_path = model_save_path_model_name + '.h5'
        self.models[model_name] = ModelCheckpoint(model_save_path,
                                             monitor=monitor,
                                             verbose=1,
                                             save_best_only=True
                                             )

