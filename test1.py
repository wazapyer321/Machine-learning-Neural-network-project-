import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#reads all the data in the csv file with pandas ( panadas make all the data into array's )
#the ", sep=";" line just replaces the ; with a comma ( , )
data = pd.read_csv("student-mat.csv", sep=";")

#here we can select the data we want from the student-mat.csv file ( all intingers ( numbers ))
#everything is a labes fx G1 G2 G3 ...
#and all the information the labes have are attributes fx first row
#Table:         G1
#attributes:    5
data = data [["G1","G2","G3","studytime","failures","absences","health","freetime"]]
#the (predict = "G3") is also knowed as labels and lables are what you are trying to get / looking for
predict = "G3"

#two arrays

#this will make a dataFrame. it will also use the data from on top
#x is also all the training data
x = np.array(data.drop([predict], 1))

#y are all the labels
y = np.array(data[predict])

#training
#we are taking the x and y from on top and splitting them into 4 diffrent arrays
#x_train is = x from on top and y_train is y from on top aswell.
#we are useing this to test ot output of the model


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


#commen this out since i dont need to run it any more i have found the best
#if you want to run it more times you can remove the commens ''' ''' and then just run it as many time as wanted
'''
#this just loops it 1000000000 ( one billion ) times to keep looking for the best model
best = 0
for _ in range(1000000000):

    #training data is for training the bot and the testing data is when useing the bot for testing ( testing is all the data and training is only a little amount)
    #Split arrays or matrices into random train and test subsets
    x_train,x_test,y_train ,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)


    #skipping the training and the saving
    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    #this is just saying if acc ( the model ) is bettere then the old one save it
    if acc > best:
        best = acc
        #useing Pickel
        #this will save a pickel file in the dir
        #write binary = wb
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear, f)
#commen  '''

#read pickle file
#read binary  = rb
pickle_in = open("studentmodel.pickle", "rb")
#will load the the model into the variable linear
linear = pickle.load(pickle_in)


#this is equel to M in the calculation ( coefficient )
#the more M's there are the mode dimensional is caluclation is fx there are 5 outputs in this one so its 5 dimensional space.
print ("coefficient: \n", + linear.coef_)
#this is equl to B in the calculation ( Intercept )
print("Intercept: \n", + linear.intercept_)

#this will take array's and will do a ton of predictions and guss on the test data ( that is not trained )
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x],y_test[x])

#saving model
#why save the model ?
#to use the best model and not to retrain your model if you are useing a lot of data it will take a while but saving it will save time.
# and if you can find a model that has a very high accuracy to use it.




#useing matplotlib to make a graph / grid
#this will show G1
#keep in mind there are 600 studients but you are not seeing all of the datapoints to the 600 student's since a lot of them are overlapping
#chose between: "G1","G2","G3","studytime","failures","absences","health","freetime"
p = 'G1'
style.use("ggplot")
#useing a scatter plot
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()


