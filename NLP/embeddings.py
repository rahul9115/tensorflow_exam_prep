import io
import tensorflow
import tensorflow.python.keras.layers
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
data,info=tfds.load("imdb_reviews",with_info=True,as_supervised=True)
train=data["train"]
test=data["test"]

training_data=[]
training_labels=[]

testing_data=[]
testing_labels=[]

for s,l in train:
    training_data.append(s.numpy().decode("utf8"))
    training_labels.append(l.numpy())

for s,l in test:
    testing_data.append(s.numpy().decode("utf8"))
    testing_labels.append(l.numpy())

vocab_size=12000
max_length=120
embedding_dim=16
trunc_type='post'
oov_tok="<OOV>"

tokenizer=Tokenizer(num_words=12000,oov_token=oov_tok)
tokenizer.fit_on_texts(training_data)
word_index=tokenizer.word_index

sequences=tokenizer.texts_to_sequences(training_data)
training_sequences=pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

sequences=tokenizer.texts_to_sequences(testing_data)
testing_sequences=pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

model=tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tensorflow.keras.layers.GlobalAveragePooling1D(),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(6,activation="relu"),
    tensorflow.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(loss="binary_crossentropy",optimizer=tensorflow.keras.optimizers.Adam(),metrics=["accuracy"])
history=model.fit(training_sequences,np.asanyarray(training_labels),validation_data=(testing_sequences,np.asanyarray(testing_labels)),epochs=10)

acc=history.history["accuracy"]
val_acc=history.history["val_accuracy"]
epochs=np.arange(1,len(acc)+1,1)
plt.figure(figsize=(10,6))
plt.plot(epochs,acc,color="blue")
plt.plot(epochs,val_acc,color="red")
plt.show()
plt.savefig("accuracy.png")


layer=model.layers[0]
weights=layer.get_weights()[0]

reverse_word_index=tokenizer.index_word

vectors=io.open("vecs.tsv",'w',encoding="utf-8")
words=io.open("words.tsv",'w',encoding="utf-8")

for i in range(1,vocab_size):
    word=reverse_word_index[i]
    word_embedding=weights[i]

    words.write(word+"\n")
    vectors.write('\t'.join([str(j) for j in word_embedding])+"\n")
vectors.close()
words.close()
