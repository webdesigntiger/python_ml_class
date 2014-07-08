# Stanford's Machine Learning by Andrew Ng - Assignments in Python format
# Felipe Ryan 2014

# The following import forces floatpoint division so 5/2 = 2.5 and not 2
from __future__ import division
import numpy as np
import math


# Helper function
def sigmoid(val):

    return 1 / (1 + math.exp(-val))


# Applies function above to every element:
def VectorizedSigmoid(matrix):
    vs = np.vectorize(sigmoid)
    return vs(matrix)


# Helper function
def score(X, y, theta):
    ssres = np.sum((X.dot(theta.T) - y.T) ** 2)
    sstot = np.sum((y.T - np.mean(y)) ** 2)

    return 1 - (ssres / sstot)


# Feature Normalization
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = np.subtract(X, mu)
    sigma = np.std(X, axis=0, ddof=1)
    return np.divide(X_norm, sigma)


# 100% done and tested
def computeCostMulti(X, y, theta):
    m = float(X.shape[0])
    J = 0
    J = np.sum(np.power(X.dot(theta) - y, 2)) / (2 * m)
    return J


# 100% done and tested
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = float(X.shape[0])
    J_hist = np.zeros((num_iters, 1))

    for i in xrange(num_iters):
        h = X.dot(theta)
        theta = theta - ((alpha / m) * (X.T.dot(h - y)))

        J_hist[i] = computeCostMulti(X, y, theta)

    return theta, J_hist


# 100% done and tested (regularized)
def lrCostFunction(theta, X, y, theLambda):

    m = X.shape[0]
    J = 0
    grad = np.zeros(theta.shape)

    J = ((-y.T.dot(np.log(VectorizedSigmoid(X.dot(theta))))) -
        ((1 - y).T.dot(np.log(1 - VectorizedSigmoid(X.dot(theta)))))) / float(m)

    J = J + (float(theLambda) / (2 * m)) * np.sum(np.power(theta[1:, :], 2))

    grad = (1.0 / m) * (X.T.dot((VectorizedSigmoid(X.dot(theta)) - y)))

    # Following line does the regularisation:
    grad[1:, :] = grad[1:, :] + (theta[1:, :] * (float(theLambda) / y.shape[0]))

    return J, grad


def gradientDescentMultiLogistic(X, y, theta, alpha, theLambda, num_iters):

    J_hist = np.zeros((num_iters, 1))

    for i in xrange(num_iters):
        (J, th) = lrCostFunction(theta, X, y, theLambda)
        theta = theta - (alpha * th)
        J_hist[i] = J

    return theta, J_hist


def oneVsAll(X, y, num_labels, theLambda=0.1, alpha=0.1, num_iters=50):

    n = X.shape[1]

    all_theta = np.zeros((num_labels, n))

    for i in xrange(num_labels):

        new_y = y == i
        new_y = new_y.astype(int)

        temp_theta = np.zeros((n, 1))

        (th, j) = gradientDescentMultiLogistic(X, new_y, temp_theta, alpha, theLambda, num_iters)

        all_theta[i] = th.T

    return all_theta


def predictOneVsAll(all_theta, X):

    probs = X.dot(all_theta.T)
    predictions = np.argmax(probs, axis=1)

    return predictions


def sigmoidGrad(val):
    return (1.0 / (1.0 + math.exp(-val))) * \
        (1 - (1.0 / (1.0 + math.exp(-val))))


# Applies function above to every element:
def VectorizedSigmoidGrad(matrix):
    vs = np.vectorize(sigmoidGrad)
    return vs(matrix)


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda=0):

    J = 0
    m = X.shape[0]

    t1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))]
    t1 = t1.reshape((hidden_layer_size, (input_layer_size + 1)), order='F')

    t2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]
    t2 = t2.reshape((num_labels, (hidden_layer_size + 1)), order='F')

    hidden = X.dot(t1.T)
    hidden = VectorizedSigmoid(hidden)
    # Add bias term:
    hidden = np.insert(arr=hidden, obj=0, values=1, axis=1)

    output = hidden.dot(t2.T)
    output = VectorizedSigmoid(output)

    yy = np.zeros((m, num_labels))
    for i in xrange(y.shape[0]):
        yy[i, (y[i] - 1)] = 1

    for i in xrange(y.shape[0]):
        J += yy[i].dot(np.nan_to_num(np.log(output[i])).T) + \
            (1 - yy[i]).dot(np.nan_to_num(np.log(1 - output[i])).T)

    J = J * (-1 / m)

    # Right till here

    t1r = t1[:, 1:]
    t2r = t2[:, 1:]

    sumt1r = np.sum(np.sum(np.power(t1r, 2), axis=1))
    sumt2r = np.sum(np.sum(np.power(t2r, 2), axis=1))

    reg = (sumt1r + sumt2r) * (theLambda / (2 * m))

    J += reg

    delta3 = output - yy
    z2 = X.dot(t1.T)
    z2 = np.insert(arr=z2, obj=0, values=1, axis=1)
    delta2 = np.multiply(delta3.dot(t2), (VectorizedSigmoidGrad(z2)))[:, 1:]

    d1 = delta2.T.dot(X)
    d2 = delta3.T.dot(hidden)

    d1 = (d1 / m) + ((theLambda / m) * (np.insert(arr=t1[:, 1:], obj=0, values=0, axis=1)))
    d2 = (d2 / m) + ((theLambda / m) * (np.insert(arr=t2[:, 1:], obj=0, values=0, axis=1)))

    grad = np.concatenate((d1.flatten('F'), d2.flatten('F')))

    return grad, J


def randInitialiseWeights(l_in, l_out, epsilon):
    w = np.random.random((l_out, l_in))
    return w * 2 * epsilon - epsilon


def gradientDescentNeuralNetwork(theta, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda, alpha, num_iters):

    J_hist = np.zeros((num_iters, 1))

    old_alpha = alpha

    for i in xrange(num_iters):
        (th, J) = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda)
        theta = theta - (alpha * th)
        J_hist[i] = J

        # An attempt at a crude adaptive alpha (learning rate)
        if i > 1:
            if J < J_hist[i - 1]:
                alpha += old_alpha
            else:
                alpha = alpha / 2

        print 'Iter %d | alpha: %f | J = %f' % (num_iters - i, alpha, J)

    return theta, J_hist


# Helper function in an attempt to use fmin_cg
def funCostNeuralNetwork(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda=0):
    (gg, jj) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda)
    return jj


# Helper function in an attempt to use fmin_cg
def funGradNeuralNetwork(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda=0):
    (gg, jj) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, theLambda)
    return gg


def predict(t1, t2, X):

    h1 = VectorizedSigmoid(X.dot(t1.T))
    h1 = np.insert(arr=h1, obj=0, values=1, axis=1)
    h2 = VectorizedSigmoid(h1.dot(t2.T))

    p = h2.argmax(axis=1)

    p = p + 1

    return p


def testNeuralNetwork():
    import scipy.io as sio
    from scipy.optimize import minimize

    # Load data
    d = sio.loadmat('ex4data1.mat')
    X = d['X']
    y = d['y']
    X = np.insert(arr=X, obj=0, values=1, axis=1)

    args = (400, 25, 10, X, y, 1)

    i_t1 = randInitialiseWeights(401, 25, 0.12)
    i_t2 = randInitialiseWeights(26, 10, 0.12)

    i_nn_params = np.concatenate((i_t1.flatten(order='F'), i_t2.flatten(order='F')), axis=0)

    print 'Done reading in data, now training Neural Network...'

    res2 = minimize(funCostNeuralNetwork, i_nn_params, args=args, method='CG', jac=funGradNeuralNetwork, options={'maxiter': 50})

    optT1 = res2['x'][0:(25 * (400 + 1))]
    optT1 = optT1.reshape((25, (400 + 1)), order='F')
    optT2 = res2['x'][(25 * (400 + 1)):]
    optT2 = optT2.reshape((10, (25 + 1)), order='F')

    p = np.asmatrix(predict(optT1, optT2, X)).T
    score = (p == y).astype(int).mean() * 100

    print res2['message']
    print 'Cost: %f' % res2['fun']
    print 'Score: %0.2f%%' % score
