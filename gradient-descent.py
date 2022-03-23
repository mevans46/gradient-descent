import numpy as np

#predict functions multiplying weights
def predict(X, w):
    return np.matmul(X, w)

#loss function
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

#gradient descent test
def gradient(X, Y, w):
    """
    w_grad =  2 * np.average(X * (predict(X, w, b) - Y))
    b_grad =  2 * np.average((predict(X, w, b) - Y))
    return (w_grad,b_grad)
    """
    return 2 * np.matmul(X.T,(predict(X,w)-Y))/X.shape[0]

#training function
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1],1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w)))
        
        w -= gradient(X,Y,w) * lr
    return w


x1,x2,x3, y = np.loadtxt("./data/data.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size),x1,x2,x3))
Y = y.reshape(-1,1)
w = train(X,Y,iterations=100000,lr=0.001)

print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))
