import os, re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN


def read_Pickle( file_name):
    with open( file_name, 'rb') as f:
        data2send = pickle.load(f)
    return data2send

def load_data():
    fullText = read_Pickle(pickle_save_path + r'\data.pickle')
    fullLabel = read_Pickle(pickle_save_path + r'\labels.pickle')
    split_index = read_Pickle(pickle_save_path + r'\data_split_index.pickle')
    print("total data len : ", len(fullText))
    print("total label len : ", len(fullLabel))
    print("index : ", split_index)

    x_train = fullText[split_index['train'][0]:split_index['train'][1]]
    y_train = fullLabel[split_index['train'][0]:split_index['train'][1]]
    # x_val = fullText[split_index['val'][0]:split_index['val'][1]]
    # y_val = fullLabel[split_index['val'][0]:split_index['val'][1]]
    x_test = fullText[split_index['test'][0]:split_index['test'][1]]
    y_test = fullLabel[split_index['test'][0]:split_index['test'][1]]
    return x_train, y_train, x_test, y_test

max_len = 50 # 최대 단어 수가 54였음 -> text당 사용할 단어 수
max_words = 10000 # 훈련시 사용할 단어 수 ->  최빈 10000개
embedding_dim = 50
pickle_save_path = r'C:\Users\codbs\PycharmProjects\tweetDisaster\data_pickle'

# 1. load data
x_train, y_train, x_test, y_test = load_data()
print ("[MODEL] data x :", x_train.shape)
print ("[MODEL] data y :", y_train.shape)

# 2. load embedding matrix
print ("[MODEL] max len : %d, max words = %d" % (max_len, max_words))
embedding_matrix = read_Pickle( pickle_save_path + r'\embedding_matrix.pickle')
print ("[MODEL] embedding matrix shape : ", embedding_matrix.shape)

# 3. model define
print ("[MODEL] 1. model define")
# 순전파 모델
model = Sequential()
model.add( Embedding( max_words, embedding_dim))
model.add( SimpleRNN(embedding_dim))
model.add( Dense(1, activation='sigmoid'))
print ( model.summary() )

# 4. model run
model.compile( optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# output save
history = model.fit( x_train, y_train, epochs=7, batch_size=128, validation_split=0.1)

# # 5. draw acc, loss
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

# 6. evaluate
loss, acc = model.evaluate(x_test, y_test)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))