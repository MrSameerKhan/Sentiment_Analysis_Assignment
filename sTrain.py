import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pickle
import json
import tensorboard

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.utils import np_utils


def sentiment_Analysis(inputFile):

    dataFrame = pd.read_csv(inputFile)

    label_Encoder = preprocessing.LabelEncoder()
    labels = label_Encoder.fit_transform(dataFrame.label)

    x_Train, x_Test, y_Train, y_Test = train_test_split(dataFrame.tweet.values, labels, stratify=labels, 
                                        random_state=42, test_size=0.15, shuffle=True)

    embedding_Index = {}
    eFile = open("embedding/glove.6B.50d.txt", encoding="utf-8")
    for sLine in tqdm(eFile):
        values = sLine.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_Index[word] = coefs
    eFile.close()

    VOCABULARY_SIZE = 2000
    MAX_LENGTH = 64

    tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
    tokenizer.fit_on_texts(list(x_Train)+list(x_Test))

    x_Train_Sequence = tokenizer.texts_to_sequences(x_Train)
    x_Test_Sequence = tokenizer.texts_to_sequences(x_Test)
    
    x_Train_Padding = sequence.pad_sequences(x_Train_Sequence, maxlen=MAX_LENGTH)
    x_Test_Padding = sequence.pad_sequences(x_Test_Sequence, maxlen=MAX_LENGTH)
    word_Index = tokenizer.word_index

    embedding_Matrix = np.zeros((len(word_Index)+1, 50))

    for word, sa in tqdm(word_Index.items()):
        embedding_Vector = embedding_Index.get(word)

        if embedding_Vector is not None:
            embedding_Matrix[sa] = embedding_Vector

    model = Sequential()
    model.add(Embedding(len(word_Index) + 1,50,weights=[embedding_Matrix],input_length=MAX_LENGTH,trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_Train_Encode = np_utils.to_categorical(y_Train)
    y_Test_Encode = np_utils.to_categorical(y_Test)

    history = model.fit(x_Train_Padding, y= y_Train_Encode,
                batch_size=15, epochs=18, verbose=1, 
                validation_data=(x_Test_Padding, y_Test_Encode))

    trained_Model = model.save("weights/sentiment.hs")

    json_Stringfy = tokenizer.to_json()

    with open("weights/sentiment.pickle", "wb") as pickleOut:
        pickle.dump(tokenizer, pickleOut, protocol=pickle.HIGHEST_PROTOCOL)

    with open("weights/sentiment.json", "w") as jsonOut:
        json.dump(json_Stringfy, jsonOut)

    print("Model has been Saved!")

if __name__ == "__main__":
    
    input_Path = "Data/train_Sentiment.csv"
    sentiment_Analysis(input_Path)


