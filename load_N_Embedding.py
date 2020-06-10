import os, re
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

max_len = 50 # 최대 단어 수가 54였음 -> text당 사용할 단어 수
max_words = 10000 # 훈련시 사용할 단어 수 ->  최빈 10000개
data_ratio_train = 0.8
data_ratio_val = 0.1
data_ratio_test = 0.1
embedding_dim = 50

def refineWords(s):
    # lowwer case
    # url, ,(comma) remove
    s = re.sub(r"http\S+|,", '', s)
    words = s.lower().split()
    # 재난 장소 표현이 많아 없애지 않음
    # stops = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if not w in stops]
    # lemmatizer : original form
    lmtzr = WordNetLemmatizer()
    lemmatized_words = [lmtzr.lemmatize(word) for word in words]
    # stemming :
    # stemmer = PorterStemmer()

    return (" ".join(lemmatized_words))

# 1. read data
print ("[PREP] 1. read data")
data_path = r"C:\Users\codbs\Yooney_TF_codes\TensorFlow_Test\DataSets\Real_or_Not_NLP_with_Disaster_Tweets.csv"
data_raw = pd.read_csv( data_path)

# 2. tokenizing
print ("[PREP] 2. tokenizing : max len : %d, max words = %d" % (max_len, max_words))

# 2.1. lower and lemdatizing
data_raw["text"] = data_raw["text"].apply(refineWords)
# test remove url  i=7610 / , for 3
# print ("original : %s" % data_raw["text"][i])
# print ("after refine&tokenized : %s" % tokenizer.sequences_to_texts( [sequences[i]])[0])

# 2.2 tokenizing
tokenizer = Tokenizer( num_words= max_words, oov_token= 0, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(data_raw["text"])
sequences = tokenizer.texts_to_sequences( data_raw["text"])
word_index = tokenizer.word_index
print ("tokenizer : %d개의 토큰을 찾음" % len(word_index))

# 2.3 padding 앞에서 부터
data = pad_sequences(sequences, maxlen=max_len)
labels = np.asarray( data_raw['target'])
print ('데이터 텐서 크기 : ', data.shape)
print ('레이블 텐서 크기 : ', labels.shape)

# 3. shuffle and set split to training, val, test
print ("[PREP] 3. shuffle and split data to train, test, val")
indices = np.arange( data.shape[0])
np.random.shuffle( indices)
data = data[indices]
labels = labels[indices]

train_index = (0, int(data.shape[0]* data_ratio_train) )
val_index = (train_index[1], int( train_index[1] + data.shape[0] * data_ratio_val))
test_index = (val_index[1], int( data.shape[0]))
print ('train ratio : ', data_ratio_train, train_index)
print ('val ratio : ', data_ratio_val, val_index)
print ('text ratio : ', data_ratio_test, test_index)

x_train = data[train_index[0]:train_index[1]]
y_train = labels[train_index[0]:train_index[1]]
x_val = data[val_index[0]: val_index[1]]
y_val = labels[val_index[0]: val_index[1]]
x_test = data[test_index[0]: test_index[1]]
y_test = labels[test_index[0]: test_index[1]]
data_split_index = { 'train' : train_index, 'val' : val_index, 'test' : test_index}

# 4. read embeding data
print ("[PREP] 4. embedding matrix create ")
embeddings_index = {}
glove_path = r"C:\Users\codbs\Yooney_TF_codes\TensorFlow_Test\DataSets\glove\glove.twitter.27B.50d.txt"
with open( glove_path, 'rt', encoding='UTF8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print ("glove : %d 개의 vector 찾음" % len(embeddings_index))


embedding_matrix = np.zeros( (max_words, embedding_dim))
for word , i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 5. save data, tokenizer, embeddingmatrix
print ("[PREP] 5. save data, tokenizer, embeddingmatrix")
# 1. data
pickle_save_path = r'C:\Users\codbs\PycharmProjects\tweetDisaster\data_pickle'
with open( pickle_save_path + r'\data.pickle', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
with open( pickle_save_path + r'\labels.pickle', 'wb') as f:
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
with open( pickle_save_path + r'\data_split_index.pickle', 'wb') as f:
    pickle.dump(data_split_index, f, pickle.HIGHEST_PROTOCOL)
# 2. tokenizer
with open( pickle_save_path + r'\tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

# 3. embedding matrix
with open( pickle_save_path + r'\embedding_matrix.pickle', 'wb') as f:
    pickle.dump(embedding_matrix, f, pickle.HIGHEST_PROTOCOL)


# # load
# with open('data.pickle', 'rb') as f:
#     data = pickle.load(f)