import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(1234)

#sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#softmax function
def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x, axis=0)

#forward pass
def forward(X,w0,b0,w1,b1,w2,b2):

    n = X.shape[1]
    X = w0.dot(X) + np.repeat(b0, n,axis =1)
    X = sigmoid(X)

    X = w1.dot(X) + np.repeat(b1, n,axis =1)
    X = sigmoid(X)

    X = w2.dot(X) + np.repeat(b2, n,axis =1)
    Y_pred = softmax(X)
    return Y_pred

#loss function for softmax output
def loss(Y,Y_pred):
    Y_pred = np.multiply(Y,Y_pred)
    y_pred = np.max(Y_pred, axis = 0)
    loss = - np.log(y_pred)
    return np.average(loss)

#in order to use in accuracy calculation - need attention
def calculate_accuracy(Y_pred,y):
    y_pred = np.argmax(Y_pred, axis=0)
    acc = np.average(np.equal(y,y_pred))
    return acc

#under construction
def backprog(loss,X,w0,b0,w1,b1,w2,b2):
    #update weights and bias using loss
    return w0,b0,w1,b1,w2,b2

def main():

    df = pd.read_csv("./data/train.csv") #absolute path olabilir
    y = np.array(df["label"])
    X = np.array(df[df.columns[1:]])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, random_state= 31)
    X_train = np.transpose(X_train)
    X_valid = np.transpose(X_valid)

    w0 = np.random.rand(50, 784) * 0.01
    b0 = np.zeros((50,1))
    w1 = np.random.rand(20, 50) * 0.01
    b1 = np.zeros((20,1))
    w2 = np.random.rand(10, 20) * 0.01
    b2 = np.zeros((10,1))

    #delete this part & update related variables when implementation finishes
    X_d = X_train[:,0:3]
    y_d = y_train[0:3]

    #take this part for initilization
    Y_d = np.zeros((10,len(y_d)))
    idx = np.arange(len(y_d))
    Y_d[y_d,idx] = 1

    first_pass = forward(X_d,w0,b0,w1,b1,w2,b2)
    accuracy = calculate_accuracy(first_pass,y_d)
    L_soft = loss(Y_d,first_pass)

    d = 7


if __name__ == "__main__":
    main()