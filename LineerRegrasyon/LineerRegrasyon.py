import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #grafik vs 

class LineerRegresyon:
    def __init__(self):
        self.coef_ = None #making coefficient definition is none and starting

    def fit(self, X, Y):#We are adding x and y values with using convertToArray method. With multiplication transpose (X.T) of these values we are calculating our multiple.
        X = self.convertToArray(X) 
        Y = self.convertToArray(Y)
        self.coef_ = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
        
    def predict(self, X):
        X = self.convertToArray(X)
        return np.dot(X,self.coef_) #We are multiplicating sended array with multiple.

    def convertToArray(self, x): # Sending back all sended values as arrays
        x = np.array(x)
        if x.ndim == 1: #If ndim and array 1(one) sized we are using reshape for make our array line and column. For Example [[2],[3],[4]] We are taking array like that.
            x = x.reshape(-1, 1)
        return x

    def result(self, X, y):# We are transforming x and y values to array.
        X = self.convertToArray(X)
        y = self.convertToArray(y)
        pred = self.predict(X) # Sending self.predict(X) and x array , multiple which is we created before multiplicate these and show the value.
        return np.mean((np.array(pred)-np.array(y))**2) #Transforming pred value and y value to array. substracting these values with each other and taking square of it , calculating the average.

dataFile = pd.read_csv("DataSet.csv") #We read our dataset.

series1 = pd.Series(dataFile["math score"]) # We take the relevant parts from our dataset.
series2 = pd.Series(dataFile["writing score"])
newFrameData = pd.concat(objs=(series1 , series2), axis=1) # We are combining the parts we take from data set. Purpose in here just using important datas not using all of them

newFrameData = newFrameData.astype(int) # We assign the data type of our data set as int
trainDataFile = newFrameData[0:int((len(newFrameData)*30)/100)] #We are choosing our learning data set's percantages as % 30 
testDataFile = newFrameData[int((len(newFrameData)*70)/100):len(newFrameData)] # Test data set ½70

X = trainDataFile["math score"] # We are getting related datas from learning set
Y = trainDataFile["writing score"]
lr = LineerRegresyon() # We set an example from our LinearResection class.
lr.fit(X,Y) # From the class example we created, we call the .fit method and send the learning data.

X = pd.Series(testDataFile["math score"]) # Making faster datas on test set and extracting datas from type of their arrays
Y = lr.predict(testDataFile["math score"]).tolist() # We send the data in our test dataset to the predicti method in the class example we created. And we turn the array that came to us into the list.
testDataFile.plot.scatter(x="math score", y="writing score") # Writings shown in the chart
plt.plot(X,Y,color="black") # The color of the line shown in the graph

result = lr.result(trainDataFile["math score"],trainDataFile["writing score"])# We send the relevant fields in our learning data set using the result method of our class and get the results related to our data set.
print(result)

plt.show() # At the last showing graphs