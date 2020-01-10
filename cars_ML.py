import tensorflow
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing

#k-Nearest Neighbors - p.1 - irregular data ( video time stamp 55:00 )
#data is is not 100% ( like missing data )

'''
Pandas will take the frist row as what the inputs are so buying = vhigh and maint is also vhigh
1: buying,maint,door,persons,lug_boot,safety,class
2: vhigh,vhigh,2,2,small,low,unacc
3: vhigh,vhigh,2,2,small,med,unacc
4: vhigh,vhigh,2,2,small,high,unacc

'''

data = pd.read_csv("car.data")
print(data.head())


#converting words to intigers fx vhigh to 3 and med to 2...
#lableencoder will take the lables and convert them into intigers  ( this is just a object )
le = preprocessing.LabelEncoder()
#this will get everything from the buying colum and take the word fx vhigh and turn it into a list & a intiger.
#this will also make a numpy array
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

print(buying)
predit = "class"

#this is a list conversion ( convert the numpy array to a list ( the zip function puts the list / arrays togeather in one line ))
x = list(zip(buying,maint,door,persons,lug_boot,safety))

y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#the n_neighbors is how many neighbors it will have ( change the number for a better accuracy )
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)

name = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("predicted: ",name[predicted[x]], "Data: ", x_test[x], "Actual: ", name[y_test[x]])
    n = model.kneighbors([x_test[x]],9, True)
    print("N: ", n)