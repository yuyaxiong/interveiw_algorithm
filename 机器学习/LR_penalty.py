import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split


def loadDataSet():
    data = load_breast_cancer()
    return data.data, data.target

# p(y=1|x) = np.exp(w*x)/(1 + np.exp(w*x))
# p(y=0|x) = 1/(1 + np.exp(w*x))
def sigmoid(x):
    return 1.0/(1 + np.exp(x))

# w = w - alpha * x * (p(y=1|x) - y)
def gradAscent(feature, label, max_iteration=100, alpha=0.001, penalty='L1'):
    # alpha 是学习率
    featureMatrix = np.mat(feature)
    labelMatrix = np.mat(label).transpose()
    m, n = featureMatrix.shape
    weights = np.mat((np.random.random((n, 1)) - 0.5)/100)
    for i in range(max_iteration):
        h = sigmoid(featureMatrix * weights)
        error = 1 - h - labelMatrix
        regular_item = 0
        if penalty == 'L1':
            regular_item = np.mat(np.where(weights > 0, 1, -1))
        elif penalty == 'L2':
            regular_item = 2 * weights
        weights = weights - alpha * (featureMatrix.transpose() * error + regular_item)
    return weights

def LR_fit(feature, label, max_iteration=100 ,alpha=0.001, penalty='L1'):
    weights = gradAscent(feature, label, max_iteration, alpha, penalty=penalty)
    return weights

def LR_predict(weights, x):
    return np.array(1 - sigmoid(x * weights))[:, 0]

if __name__ == '__main__':
    for penalty in [None, 'L1', 'L2']:
        tmp_test, tmp_train = [], []
        for j in range(100):
            feature, label = loadDataSet()
            train_x, test_x, train_y, test_y = train_test_split(feature, label)
            weights = LR_fit(train_x, train_y, max_iteration=1000, penalty=penalty)
            train_prob = LR_predict(weights, train_x)
            test_prob = LR_predict(weights, test_x)
            auc_train = roc_auc_score(train_y, train_prob)
            auc_test = roc_auc_score(test_y, test_prob)
            tmp_test.append(auc_test)
            tmp_train.append(auc_train)
        print(penalty, sum(tmp_test)/len(tmp_test), sum(tmp_train)/len(tmp_train))