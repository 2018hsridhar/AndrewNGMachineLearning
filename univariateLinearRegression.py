# Descript : Given a range of home sizes ( square footage ) and a range of prices ( dollars )
# Let us fit a BLOF : Best Line of Fit
# Supervised learning : univariate ( one variable ) linear regression

# Commit log :
# (A) Gotta learn sklearn
# (B)
# (C)
# (D)

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# size(testData) = 25% * size(trainData)
def getTestData():
    colList = ['house size','median price']
    testData = pd.DataFrame(columns=colList)
    low = mid - 500
    high = mid + 500
    sqFeetCol = pd.DataFrame(np.random.randint(low,high,size=(25, 2)), columns=colList)
    testData = pd.concat([trainingData, sqFeetCol], ignore_index=True)
    return testData
    
# Step 1. Generate our training data ( our features ) 
# Generate ranges of data ( with low,mid,high ) as we proceed
colList = ['house size','median price']
trainingData = pd.DataFrame(columns=colList)
testingData = getTestData()


for mid in range(500,9500,1000):
    low = mid - 500
    high = mid + 500
    sqFeetCol = pd.DataFrame(np.random.randint(low,high,size=(100, 2)), columns=colList)
    trainingData = pd.concat([trainingData, sqFeetCol], ignore_index=True)
    X_train = pd.DataFrame(trainingData['house size'])
    Y_train = pd.DataFrame(trainingData['median price'])
    
    testData = getTestData()
    X_test = pd.DataFrame(testData['house size'])
    # df = pd.DataFrame([series])
    Y_test = pd.DataFrame(testData['median price'])

    # Gotta learn sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X_train,Y_train)

    # Use model to make predictions ( on testing set )
    Y_Pred = regr.predict(X_test)

    # Get model information and ML model metrics
    featureWeights = pd.DataFrame(zip(X_test.columns, regr.coef_))
    biasTerm = regr.intercept_
    print("Feature weights = ")
    print(featureWeights)
    print("Bias term = " + str(biasTerm))
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_Pred))

    # Plot our test datasets ( X_test,Y_test)
    # Plot the LOBF : Line-of-Best-Fit
    # Visually compare the deltas between the predictions versus the ground truth labels.
    plt.scatter(X_test, Y_test, color="black")
    plt.scatter(X_test, Y_Pred, color="blue",linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()








