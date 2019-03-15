import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

def loadDataSet():
    data = load_breast_cancer()
    return data.data, data.target

# p(y=1|x) = np.exp(w*x)/(1 + np.exp(w*x))
# p(y=0|x) = 1/(1 + np.exp(w*x))
def sigmoid(x):
    return 1.0/(1 + np.exp(x))

# w = w - alpha * x * (p(y=1|x) - y)
def gradAscent(feature, label, max_iteration=100, alpha=0.001):
    # alpha 是学习率
    featureMatrix = np.mat(feature)
    labelMatrix = np.mat(label).transpose()
    m, n = featureMatrix.shape
    weights = (np.random.random((n, 1)) - 0.5)/100
    for i in range(max_iteration):
        h = sigmoid(featureMatrix * weights)
        error = 1 - h - labelMatrix
        weights = weights - alpha * featureMatrix.transpose() * error
    return weights

def LR_fit(feature, label, max_iteration=100 ,alpha=0.001):
    weights = gradAscent(feature, label, max_iteration, alpha)
    return weights

def LR_predict(weights, x):
    return np.array(1 - sigmoid(x * weights))[:, 0]

if __name__ == '__main__':
    feature, label = loadDataSet()
    weights = LR_fit(feature, label, max_iteration=1000)
    y_prob = LR_predict(weights, feature)
    auc = roc_auc_score(y_prob, label)
    print(auc)


