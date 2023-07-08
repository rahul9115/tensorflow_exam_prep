import tensorflow as tf
import numpy as np

inputs=np.array([[1.0,3.0,4.0,2.0]])
inputs=tf.convert_to_tensor(inputs)
print("inputs to softmax: ",inputs.numpy())
l=[]
sum1=0
for i in inputs[0]:
    sum1+=np.exp(i)
for j in inputs[0]:
    l.append(np.exp(j)/sum1)
print("output: ",l)

outputs=tf.keras.activations.softmax(inputs)
print("output of softmax function: ",outputs.numpy())

sum=tf.reduce_sum(outputs)
print("sum of outputs: ",sum)

print(": ",{np.argmax(outputs)})
