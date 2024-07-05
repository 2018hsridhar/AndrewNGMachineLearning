# Descript : Given a range of home sizes ( square footage ) and a range of prices ( dollars )
# Let us fit a BLOF : Best Line of Fit
# Supervised learning : univariate ( one variable ) linear regression
# Business use case : Assist REAs - real estate agents - in assessing home values quickly and minimize manual human work in assessing. Make markets more liquid.
# Refactor as we go for better methods :-) 
# await micropip.install("scipy") in Python

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d


# Step 1. Generate our training data ( our features ) 
# Generate ranges of data ( with low,mid,high ) as we proceed
featureList = ['house size']
targetVariables = ['median price']
                   
trainingData = getTrainingData()
testData = getTestData(trainingData)

X_train = pd.DataFrame(trainingData[featureList])
Y_train = pd.DataFrame(trainingData[targetVariables])

X_test = pd.DataFrame(testData[featureList])
Y_test = pd.DataFrame(testData[targetVariables])


# size(testData) = 25% * size(trainData)
def getTestData(trainingData):
    colList = ['house size','median price']
    testData = pd.DataFrame(columns=colList)
    lower = 5
    upper = 95
    step = 10
    rangeDelta = 5
    pageSize = 100
    numCols = 2
    for mid in range(lower, upper,step):
        low = mid - rangeDelta
        high = mid + rangeDelta
        sqFeetCol = pd.DataFrame(np.random.randint(low,high,size=(pageSize, numCols)), columns=colList)
        testData = pd.concat([trainingData, sqFeetCol], ignore_index=True)
    return testData

# Training data 
def getTrainingData():
    colList = ['house size','median price']
    lower = 5
    upper = 95
    step = 10
    rangeDelta = 5
    pageSize = 100
    numCols = 2
    trainingData = pd.DataFrame(columns=colList)
    for mid in range(lower, upper,step):
        low = mid - rangeDelta
        high = mid + rangeDelta
        sqFeetCol = pd.DataFrame(np.random.randint(low,high,size=(pageSize, numCols)), columns=colList)
        trainingData = pd.concat([trainingData, sqFeetCol], ignore_index=True)
    return trainingData

def main():

    # Gotta learn sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X_train,Y_train)

    # Use model to make predictions ( on testing set )
    Y_Pred = regr.predict(X_test)

    # Get model information and ML model metrics
    # featureWeights = pd.DataFrame(zip(X_test.columns, regr.coef_))
    # biasTerm = regr.intercept_
    # print("Feature weights = ")
    # print(featureWeights)
    # print("Bias term = " + str(biasTerm))
    # print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_Pred))

    # Plot our test datasets ( X_test,Y_test)
    # Plot the LOBF : Line-of-Best-Fit
    # Visually compare the deltas between the predictions versus the ground truth labels.
    # plt.scatter(X_test, Y_test, color="black")
    # plt.scatter(X_test, Y_Pred, color="blue",linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

    # Solve for cost function J(W,b) : it's really a scaled version of MSE in the hiding
    # Here, cost function based on number of training examples ( versus test examples )?
    # print("Sovling current cost function for linear regression model")
    # m = X_train.shape[0]
    # Y_train = pd.DataFrame(trainingData[targetVariables])
    # deltaTrainPred = (Y_train - Y_pred)
    # costValue = (1/(2*m)) * (((deltaTrainPred).pow(2)).sum())
    # mse = mean_squared_error(Y_train, Y_pred)
    # mae = mean_absolute_error(Y_train, Y_pred)
    # print("Most current cost value ( MSE ) = {}".format(costValue))
    # print("Mean Squared Error = {}".format(mse))
    # print("Mean Absolute Error = {}".format(mae))

    # Solve for a cost function visualization ( vary the weight and the bias terms )
    # Easier in lower dimensional space ( 3D visual ) -> higher is hard
    # (A) Get YTrain ( on singel example ) ( best weight,bias combo )
    # (B) Get the 100x100 examples ( with Y_pred)
    # (C) Solve for the deltas
    # Y_pred = regr.predict(X_train)
    
    weightFrame = pd.DataFrame(pd.Series(np.arange(-10,10,1),name='weight'))
    biasFrame = pd.DataFrame(pd.Series(np.arange(-10,10,1),name='bias'))
    weightBias = pd.merge(weightFrame, biasFrame, how ="cross")
    weightBias['cost'] = weightBias.apply(costFunc, axis=1) 

    # fig, ax = plt.subplots()
    # X, Y = np.meshgrid(weightBias['weight'], weightBias['bias'])
    # Z = weightBias['cost']
    # CS = ax.contour(X,Y,Z)
    # ax.clabel(CS, inline=True, fontsize=10)
    # ax.set_title('Contour plot of cost function over weight-bias cartesian product')

    fig = plt.figure()
     
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
     
    # defining axes
    # z = np.linspace(0, 1, 100)
    # x = z * np.sin(25 * z)
    # y = z * np.cos(25 * z)
    z = weightBias['cost']
    x = weightBias['weight']
    y = weightBias['bias']
    c = x + y
    ax.scatter(x, y, z, c = c)
     
    # syntax for plotting
    ax.set_title('3d Scatter plot of cost value W.R.T  geeks for geeks')
    plt.show()

# lambda function capture state better?
def costFunc(entry):
    weight = entry['weight']
    bias = entry['bias']
    Y_pred = X_train.apply(lambda x: (x*weight) + bias)
    mse = mean_squared_error(Y_train, Y_pred)
    return mse
            
            
    


    
    

main()
