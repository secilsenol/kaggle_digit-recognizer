import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(1234)

#sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#softmax function
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

#forward pass - bir veya daha fazla instance için çalışıyor
def forward(X,y,w0,b0,w1,b1,w2,b2):
    X = X.dot(w0) + b0
    X = sigmoid(X)
    X = X.dot(w1) + b1
    X = sigmoid(X)
    X = X.dot(w2) + b2
    X = softmax(X)

    #loop dışında bir çözüm??
    loss = np.zeros(len(y))
    for i in range(len(y)):
        loss[i] = - np.log10(X[i][y[i]])
    return loss

#under construction
def backprog(loss,X,w0,b0,w1,b1,w2,b2):
    #update weights and bias using loss
    return w0,b0,w1,b1,w2,b2

#make prediction using the updated weights and bias
def predict(X_new,w0,b0,w1,b1,w2,b2):
    X = X_new.dot(w0) + b0
    X = sigmoid(X)
    X = X.dot(w1) + b1
    X = sigmoid(X)
    X = X.dot(w2) + b2
    X = softmax(X)
    #if statement için çözüm??
    if len(X.shape) == 2:
        y_pred = np.argmax(X, axis = 1)
    else:
        y_pred = np.argmax(X)

    return y_pred


def main():

    df = pd.read_csv("./data/train.csv") #absolute path olabilir
    y = np.array(df["label"])
    X = np.array(df[df.columns[1::]])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, random_state= 31)

    w0 = np.random.rand(784, 50) * 0.01
    b0 = np.zeros((50))
    w1 = np.random.rand(50, 20) * 0.01
    b1 = np.zeros((20))
    w2 = np.random.rand(20, 10) * 0.01
    b2 = np.zeros((10))

    X_d = X[0:3]

    deneme = forward(X_d,y[0:3],w0,b0,w1,b1,w2,b2)
    sonuc = predict(X_d, w0, b0, w1, b1, w2, b2)

    d = 7


if __name__ == "__main__":
    main()