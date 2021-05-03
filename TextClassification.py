import tensorflow as td
from tensorflow import keras
import numpy as np

# load in movie review
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000) # only take 10000 most frequenly used words

#print(train_data[0])

# get the dictionary for word mapping so we can transfer review number back to word
word_index = data.get_word_index()
# deconpose the word to key(k) and value(v) so that the first 4 key are reserved for special characters
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0         # padding: make all review to the same length
word_index["<START>"] = 1       # start
word_index["<UNK>"] = 2         # unknown
word_index["<UNUSED>"] = 3      # unused

# we only have word as key and number as value, now we swap them 
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# trim data to have max 250 length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], padding = "post", maxlen = 250)

def decode_review(text):  # text is a list
    return " ".join([reverse_word_index.get(word, "?") for word in text]) # "?" if the value did not map to any word

#print(decode_review(train_data[0]))

model = keras.Sequential()
# randomly create 10000 word vectors(each of 16D) input list will grab corresponding vector and pass it to next layer
# for example, [1,5, 11, 255] grab number 1 5 11 255 vector and pass
# if we have "great" and "good" as 2 vectors, we want them to get close to each other
# it will return 16 dimensional data
model.add(keras.layers.Embedding(10000, 16))
# it takes 16 dimensional data and reduced to 1D 
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))  # final output review(good or bad) 0 or 1

#model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# use validation to varify our model while training it
x_val = train_data[:10000] # had 25000 first 10000 data points are validation set
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# batch_size meaning how many review we do each time
fitModel = model.fit(x_train, y_train, epochs = 40, batch_size = 512, validation_data = (x_val, y_val), verbose = 1)

results = model.evaluate(test_data, test_labels)
print(results) # gives [loss, accuracy]

# see how the model performe
test_review = test_data[0]
prediction = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(np.round(prediction[0])))
print("Actual: " + str(test_labels[0]))