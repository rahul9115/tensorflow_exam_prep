import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

with open("poetry.txt","r") as file:
    data=file.read()
data=data.lower()
data=data.split("\n")

tokenizer=Tokenizer(num_words=len(data))
tokenizer.fit_on_texts(data)
word_index=tokenizer.word_index
total_words=len(word_index)+1

input_sequences=[]
for i in range(len(data)):
    sequences=tokenizer.texts_to_sequences([data[i]])[0]
    for j in range(1,len(sequences)):
        input_sequences.append(sequences[:j+1])
max_sequence_len=max([len(i) for i in input_sequences])
input_sequences=pad_sequences(input_sequences,maxlen=max_sequence_len,padding="pre")
x=input_sequences[:,:-1]
y=input_sequences[:,-1]
y=tensorflow.keras.utils.to_categorical(y,num_classes=total_words)

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Embedding(total_words, 64, input_length=max_sequence_len - 1),
    tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(20)),
    tensorflow.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x, y, epochs=500)


plt.plot(history.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.show()


seed_text = "Laurence went to Dublin"
next_words = 100

for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    probabilities = model.predict(token_list)
    predicted = np.argmax(probabilities, axis=-1)[0]


    if predicted != 0:
        output_word = tokenizer.index_word[predicted]

    seed_text += " " + output_word

    print(seed_text)

seed_text = "Laurence went to Dublin"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    probabilities = model.predict(token_list)
    choice=np.random.choice([1,2,3])
    predicted = np.argsort(probabilities)[-choice]

    if predicted != 0:
        output_word = tokenizer.index_word[predicted]

    seed_text += " " + output_word

    print(seed_text)





