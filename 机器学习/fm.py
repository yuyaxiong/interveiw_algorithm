import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score


def load_data():
    data = load_breast_cancer()
    return data.data, data.target


class FM(object):
    def __init__(self, feature_n, hidden_n):
        """
        :param feature_n:特征个数
        :param hidden_n: 隐向量维度
        w0: FM的bais项
        w: 一阶权重项 shape(feature_n)
        v: 二阶交叉项的隐向量 shape（feature_n, hidden_n)
        :return:
        """
        self.feature_n = feature_n
        self.hidden_n = hidden_n
        self.w0 = np.random.random()
        self.w = np.array([np.random.random() for _ in range(feature_n)])
        self.v = np.array([[np.random.random() for _ in range(hidden_n)] for _ in range(feature_n)])
        self.scaler = MinMaxScaler()

    def predict(self, x):
        """
        :param x:shape(1,feature_n)
        :return:
        """
        x = self.scaler.transform(x)
        return 1.0 - self.proba_0(x)

    def predict_all(self, x):
        """
        :param x: shape(sample_n, feature_n)
        :return:
        """
        prob_list = []
        for i in range(x.shape[0]):
            prob_list.append(self.predict(x[i, :].reshape(1, x.shape[1])))
        return np.array(prob_list)

    def proba_0(self, x):
        """
        :param x: shape(1, feature_n)
        :return:
        """
        item1 = np.dot(np.mat(x), np.mat(self.w).T)[0, 0]
        item2 = self.cross_item(x)
        item3 = self.w0
        return self.sigmoid(item1 + item2 + item3)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cross_item(self, x):
        """
        x: shape(1, feature_n)
        """
        inner_1 = np.mat(x) * np.mat(self.v)
        inner_2 = np.mat(x * x) * np.mat(self.v * self.v)
        cross_value = np.sum(np.multiply(inner_1, inner_1) - inner_2) / 2.0
        return cross_value

    def fit(self, x, y, iter_num, alpha):
        """
        :param x: shape(sample_n,feature_n)
        :param y: shape(sample_n,1)
        :param iter_num: 迭代次数
        :param alpha: 学习率
        :return:
        """
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        for iter in range(iter_num):
            for i in range(x.shape[0]):
                sample_x = x[i, :].reshape(1, 30)
                self.update_gradient_by_sgd(sample_x, y[i], alpha)

    def update_gradient_by_sgd(self, x, y, alpha):
        """
        :param x: shape(1,feature_n)
        :param y: int
        :param alpha: float 学习率
        :return:
        """
        prob_0 = self.proba_0(x)
        self.w0 = self.w0 - alpha * (prob_0 - y)
        self.w = self.w - alpha * (prob_0 - y) * x[0, :]
        for f in range(self.feature_n):
            self.v[f, :] = self.v[f, :] * x[0, f]
        item1_1 = np.sum(self.v, axis=0)
        for f in range(self.feature_n):
            item1 = x[0, f] * item1_1
            item2 = self.v[f, :] * (x[0, f] ** 2)
            # alpha * (prob_0 - y) * x[0, f] 这一项注意是根据sigmoid作为f(x),利用极大似然估计算出来的
            delta = alpha * (prob_0 - y) * (item1 - item2)
            self.v[f, :] = self.v[f, :] - delta
        logloss = - (y * np.log(prob_0) + (1 - y) * np.log(1 - prob_0))
        # print(np.sum(self.w), np.sum(self.v), logloss)
        # print(logloss)


if __name__ == "__main__":
    x, y = load_data()
    print(x.shape)
    print(y.shape)
    # for i in range(50):
    #     fm = FM(x.shape[1], 3)
    #     fm.fit(x, y, 2, 0.0001)
    #     prob = fm.predict_all(x)
    #     auc = roc_auc_score(y, prob)
    #     print(auc)

    fm = FM(x.shape[1], 3)
    fm.fit(x, y, 2, 0.0001)
    prob = fm.predict_all(x)
    auc = roc_auc_score(y, prob)
    print(auc)
