import numpy as np
x1 = [10, 12, 13, 0, 2, 0]
x2 = [100, 120, 146, 146, 147.46, 147.46]

Y = np.array([0, 2, 4, 4, 5, 4])
Y = np.reshape(Y, (len(Y), 1))
print(Y)

import tensorflow as tf
# I love to_categorical
to_categorical = tf.keras.utils.to_categorical

y1 = to_categorical(Y)
print(y1)

"""
The main idea is to see the most important input, that does not alineate easily with others.

For this, I'll set the input like they are, to the model. By the way, the model will have to predict a uniform output (like [1,1,1,1])

This way I will be able to detect the abnormal inputs that do not align with others easily.

And these words, the model shouldn't be trained too much.
"""

dump_output = np.ones((len(x1), 1))
print(dump_output)

""" Data treatment"""
X= np.array([])

for i in range(len(x1)):
  delete_me = np.array([x1[i], x2[i]])
  X = np.append(X, delete_me)
X = np.reshape(X,(len(x1), 2))
print(X)

print(X)

len(y1[0])

normalized_data = np.array([])
for sample in X:
  sample_data = (sample[0])/(np.exp2(sample[1]))

  normalized_data = np.append(normalized_data, sample_data)
print(normalized_data)

X.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


cleaner_wrasse = Sequential()

cleaner_wrasse.add(Dense(128, activation='relu'))
# Example input shape: (timesteps, features)

cleaner_wrasse.add(Dense(1, activation='relu'))

cleaner_wrasse.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
cleaner_wrasse.build(input_shape=(None, 2, 1))
cleaner_wrasse.summary()

cleaner_wrasse.fit(X, dump_output, epochs=1) # Add epochs for training

X1 = np.array([X[1]])
cleaner_wrasse.predict(X1)

"""Fine so we got the error of the model according to the original input."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


whale = Sequential()

whale.add(LSTM(128, activation='relu'))

whale.add(Dense(1, activation='relu'))

whale.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
whale.build(input_shape=(None, 2, 1))
whale.summary()

Y[1]

x_for_whale = np.ones(X.shape)
y_for_whale = np.ones(Y.shape)
#echauffement (ouh, ouh...)
whale.fit(x_for_whale,y_for_whale)

"""Let's train our model"""
iteration = 5
for j in range(iteration):
 for i in range(len(X)):
  X1 = np.array([X[i]])
  O1 = cleaner_wrasse.predict(X1) #output 1
  input1 = np.array([O1])
  input1 = np.reshape(input1, (1,2,1,1))
  O2 = np.array([Y[i],Y[i]])
  O2 = np.reshape(O2, (1,1,2))
  print(O2.shape)
  print(input1.shape)

  print(O2,input1)
  whale.fit(input1, O2, epochs=1)

""" Enfin !!!!!!! ce n'est qu'à 3h du matin le 05/10/2025 que cela marche (commencement 22h)"""
