import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from svm import sgd
from sklearn.metrics import accuracy_score

cwd = os.getcwd()
base = os.path.join(cwd.replace('/src', ''), 'data')

data = pd.read_csv(os.path.join(base, 'cancer.csv'))

mapped = {'M': 1, 'B': -1}

data['diagnosis'] = data['diagnosis'].map(mapped)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

Y = data.loc[:,'diagnosis']
X = data.iloc[:,1:]
X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

X.insert(loc=len(X.columns), column='intercept', value=1)

X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=42)

C = 10000
learning_rate = 0.000001

W = sgd(X_train.to_numpy(), Y_train.to_numpy(), C, learning_rate)
Y_test_predicted = np.array([])
for i in range(X_test.shape[0]):
    yp = np.sign(np.dot(W, X_test.to_numpy()[i]))
    Y_test_predicted = np.append(Y_test_predicted, yp)

print('Accuracy: ', accuracy_score(Y_test.to_numpy(), Y_test_predicted))
