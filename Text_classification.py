import tensorflow as tf
from tensorflow import keras
#this will only work with numpy 1.16.1 ( to install numpy 1.16.1 : pip install numpy==1.16.1 )
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), ( test_data, test_labels) = data.load_data(num_words=880000)
#print(train_data[0])

word_index = data.get_word_index()
#this is used to make the readble for humans since all words are trasfered into intersers
word_index = {k:(v+3) for k , v in word_index.items()}

#padding
word_index["<PAD>"] = 0
#start
word_index["<START>"] = 1
#unknowen
word_index["<UNK>"] = 2
#not used
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value ) in word_index.items()])

#making putting the data at 250 words
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])
#print(decode_review(test_data[0]))
'''
#model
model = keras.Sequential()
#adding the layer with the model.add method
model.add(keras.layers.Embedding(880000, 16))
#embedding word
#create 10000 word vectors and have a 16 dimensional space for every number ( word )
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
#final output is if the review is good or bad
model.summary()

#loss function will calculate how off it is from 0 or 1
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


#Valadation data ( refreshing new data )
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitmodel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
#batch_size is how many movie review we will load in at the time.

result = model.evaluate(test_data, test_labels)

model.save("model.h5")
'''
#review_encode function
def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

#to load in the model.h5 ( the training data )
model = keras.models.load_model("model.h5")

with open("KommenTar_text.txt", encoding="utf-8") as f:
    for line in f.readlines():
        #this is to replace al the , "" [] . :
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
        #if the prediction is closes the 1 then the review is possitive and if its closes to 0 its negative
'''
#this just print out the information
test_review = test_data[0]
predict = model.predict([test_review])
print("review: ")
print(decode_review(test_review))
print("prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(result)

'''