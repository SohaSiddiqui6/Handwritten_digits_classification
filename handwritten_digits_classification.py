import tensorflow
import keras

#28x28 images of handwritten digits from 0-9
mnist = keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

#scaling values of tensor between zero and one

x_train = keras.utils.normalize(x_train,axis=1)
x_test = keras.utils.normalize(x_test,axis=1)

#visualizing training data
import matplotlib.pyplot as plt

plt.imshow(x_train[0])
print(x_train[0])
plt.show

#training neural network model

model = keras.models.Sequential()

#input layer
model.add(keras.layers.Flatten())

#hidden layer
model.add(keras.layers.Dense(100, activation='relu'))

#shidden layer
model.add(keras.layers.Dense(100, activation='relu'))

#output layer
model.add(keras.layers.Dense(10, activation='softmax'))

#compiling ANN
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#fitting training set
model.fit(x_train, y_train, epochs=10)

#loss and accuracy

loss_val,acc_val = model.evaluate(x_test,y_test)

print(loss_val,acc_val)


#saving model
model.save('handwritten_number.model')

#loading model
new_model = keras.models.load_model('handwritten_number.model')


#predicting testing data
pred = new_model.predict([x_test])
print(pred)


#visualizing test data then predicting
import numpy as np
plt.imshow(x_test[0])
print(np.argmax(pred[0]))
