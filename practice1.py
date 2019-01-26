import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sklearn.linear_model

# Read in values from csv file and load into DataFrame type from pandas
trainSet = pd.read_csv(r"Titanic\Data\train.csv")
#print('Max',trainSet['Age'].max())

# Drop vals
# trainSet.dropna(inplace=True)

# Replace vals with mean
trainSet.fillna(trainSet.mean(), inplace=True)

# Male == 1, Female == 0

# Map version
#trainSet['Sex'] = map(lambda x: 1 if x == 'male' else 0, trainSet['Sex'])

# List comprehension version
trainSet['Sex'] = [1 if i == 'male' else '0' for i in trainSet['Sex'] ]


X = np.array(trainSet[['Age','Sex','Pclass']]).reshape(-1,3)

#for item in X:
#        print(item)
y = trainSet['Survived']

model = sklearn.linear_model.LogisticRegression()
model.fit(X,y)

# File with all test data
testSet = pd.read_csv(r"Titanic\Data\test.csv")
testSet.fillna(trainSet.mean(), inplace=True)
testSet['Sex'] = map(lambda x: 1 if x == 'male' else 0, testSet['Sex'])
X_test = np.array(testSet[['Age','Sex','Pclass']]).reshape(-1,3)

dataList = []
for item in X_test:
        res = model.predict(item.reshape(1,-1))
        #print(res)
        dataList.append(*res) 
#print(len(X_test))

dataDict = {'PassengerId': testSet['PassengerId'], 'Survived': dataList}

dataDF = pd.DataFrame.from_dict(dataDict)

dataDF.to_csv(path_or_buf=r"Titanic\Data\predictions.csv",mode='w',index=False)