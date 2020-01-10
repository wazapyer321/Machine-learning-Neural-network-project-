import tensorflow as tf
#keras is a api for tensorflow ( write less code )
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#data set
data = keras.datasets.fashion_mnist

(train_images, train_labels), ( test_images, test_labels) = data.load_data(num_words=10000)
#we are useing 10 neurons 1 for each obejct
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([
    #its the input and it flattens the data.
    keras.layers.Flatten(input_shape=(28,28)),
    #we got 128 amount of neurons
    #relu stands for Rectifier which is a math function like sigmoid
    keras.layers.Dense(128,activation="relu"),
    #the softmax is when its trying to see if its 80% this or 60% something else.
    #thise are called " loss functions ( loss function learns to reduce the error in prediction )"
    #the 10 is the amout of neurons
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


#epoch will randomly pick labels and images and feed that to the neural network, the amount of epochs is how many times we feed the same image
#we do this because the order the images are comming in will have an influsense on how the parameters are tweaked
#more epochs does not nessarlery mean it will incress in accuracy
model.fit(train_images, train_labels, epochs=7)


test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("accuracy: ",test_acc)

#show image with matplotlib
#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()

#prediction
#when doing prediction's you just do model.predict( list or numpy array )
prediction = model.predict(test_images)
#this just makes a graph to see the prediction makes it easyer to read for a human
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    # argmax takes the largest value and findes the index of it. the prediction and output it
    plt.title("prediction " + class_names[np.argmax(prediction[i])])
    plt.show()
