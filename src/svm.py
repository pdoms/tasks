import numpy as np
from math_ops import dot
from sklearn.utils import shuffle


def hinge_loss(W, X, Y, C):
    N = X.shape[0]
    distances = 1 -Y * (np.dot(X, W))
    distances[distances < 0] = 0
    h_loss = C * (np.sum(distances) / N)
    return 1/2 * np.dot(W, W) + h_loss

def _hinge_loss(W,X,Y,C):
    N = len(X)
    distances = Y *(dot(X, W))
    distances = [0 if d < 0 else 1 for d in distances]  
    h_loss = C * (sum(distances) / N)
    return 1/2 * dot(W, W) + h_loss

def hinge_loss_gradient(W, X_batch, Y_batch, C):
    if type(Y_batch) == np.float64 or type(Y_batch) == np.int64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0,d) == 0:
            di = W
        else:
            di = W - (C * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)
    return dw

def sgd(features, outputs, C, learning_rate):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshild = 0.01
    for epoch in range(1, max_epochs):
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = hinge_loss_gradient(weights, x, Y[ind], C)
            weights = weights - (learning_rate * ascent)
        if epoch == 2 ** nth or epoch == max_epochs -1:
            cost = hinge_loss(weights, features, outputs, C)
            print(f"Epoch: {epoch} | Cost: {cost}")
            if abs(prev_cost - cost) < cost_threshild*prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights