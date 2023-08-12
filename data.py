from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

class Data(object):
    def __init__(self, data, y=None, label2idx=None, idx2label=None, pad_cut_seq=None, word_count=None):
        self.tokenizer = Tokenizer(filters='–"—#$%&;()+,-./:;<=>@[\\]^_`{|}~\t\n\r',
                              lower=True,
                              split=' ',
                              char_level=False)
        self.pad_cut_seq = pad_cut_seq
        self.word_count = word_count
        self.data = data
        self.y = y
        self.idx2label = idx2label
        self.label2idx = label2idx
        self.labels_num = len(idx2label.keys())


    def fit(self):
        # словарь формируется на основе встречаемости слов в тексте (больше->меньше)
        self.tokenizer.fit_on_texts(self.data)

        # padding
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

    def shape(self):
      return self.data.shape

    # урезает токенизированную последовательность сзади
    def cut(self, seqs):
        cut_seq = []
        for i in seqs:
            lens = len(i)
            if lens > 500:
                cut_seq.append(i[lens - 500:])
            else:
                cut_seq.append(i)
        return cut_seq

    def preprocess(self, text=None):

        if text is not None:
            seqs = self.tokenizer.texts_to_sequences(text)
            cut_seq = self.cut(seqs)
            pad_cut_seq = pad_sequences(cut_seq)
            return pad_cut_seq

        seqs = self.tokenizer.texts_to_sequences(self.data)
        # урежем последовательности, берём за основу гипотезу, что самое главное в конце
        cut_seq = self.cut(seqs)
        # приводим последовательности к одной длине, заполняя нулями
        self.pad_cut_seq = pad_sequences(cut_seq)
        # количество слов в словаре
        self.word_count = len(self.tokenizer.index_word)


