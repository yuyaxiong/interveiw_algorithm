from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score


def load_data():
    data = load_breast_cancer()
    return data.data, data.target


class FFM(object):
    def __init__(self, feature_n, hidden_n, field_n, feature_field_dict):
        """
        :param feature_n:特征维度个数
        :param hidden_n: 隐向量维度个数
        :param field_n: FFM中的特征域的维度个数
        :param feature_field_dict: 特征和对应域的字典，key为特征id，value为对应的域id
        """
        self.feature_n = feature_n
        self.hidden_n = hidden_n
        self.field_n = field_n
        self.w = np.random.rand(1, self.feature_n)
        self.w0 = np.random.random()
        self.v = np.random.rand(self.feature_n, self.field_n, self.hidden_n)
        self.scaler = MinMaxScaler()
        self.feature_filed_dict = feature_field_dict
        self.gradients = np.zeros((self.feature_n, self.field_n, self.hidden_n))
        self.ada_grad_lambda = 0.9

    def fit(self, x, y, eta, alpha, iter_n):
        """
        :param eta: 梯度更新中原梯度的占比
        :param x: shape(sample_n, feature_n)
        :param y: shape(1, feature_n)
        :param alpha: float 学习率
        :param iter_n: int 迭代次数
        :return:
        """
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        for iter in range(iter_n):
            for i in range(x.shape[0]):
                sample_x = x[i, :].reshape(1, x.shape[1])
                self.update_gradient(sample_x, y[i], eta,alpha)

    def update_gradient(self, x, y, eta,alpha):
        """
        :param eta: float 梯度更新中原梯度的占比
        :param x: shape(1,feature_n)
        :param y: int
        :param alpha: float 学习率
        :return:
        """
        prob_0 = self.prob_0(x)
        self.w0 = self.w0 - alpha * (prob_0 - y)
        self.w = self.w - alpha * (prob_0 - y) * x[0, :]
        for j1 in range(self.feature_n):
            for j2 in range(j1+1, self.feature_n):
                j1_field = self.feature_filed_dict[j1]
                j2_field = self.feature_filed_dict[j2]
                # 这里的后面一项梯度项是自己基于极大似然估计在fx=sigmoid和y=[0,1]的情况下，计算出来的损失项k=(prob_0 -y),等价于论文上的形式，
                # 论文上使用的是logistic损失求导得出的递推公式，logistics损失就是y=[1,-1]时候计算的交叉熵损失
                g_j1_f2 = self.v[j1, j2_field, :] * eta - (prob_0 - y) * self.v[j2, j1_field, :] * x[0, j1] * x[0, j2]
                g_j2_f1 = self.v[j2, j1_field, :] * eta - (prob_0 - y) * self.v[j1, j2_field, :] * x[0, j2] * x[0, j1]
                self.gradients[j1, j2_field, :] += g_j1_f2 ** 2
                self.gradients[j2, j1_field, :] += g_j2_f1 ** 2
                self.v[j1, j2_field, :] = self.v[j1, j2_field, :] - (alpha/self.gradients[j1, j2_field, :]) * g_j1_f2
                self.v[j2, j1_field, :] = self.v[j2, j1_field, :] - (alpha/self.gradients[j2, j1_field, :]) * g_j2_f1
        # print(self.w0, np.sum(self.w), np.sum(self.v))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, x):
        """
        :param x: shape(1, feature_n)
        :return:
        """
        return 1 - self.prob_0(x)

    def predict_all(self, x):
        """
        :param x: x shape(sample_n, feature_n)
        :return:
        """
        prob_list = []
        for i in range(x.shape[0]):
            prob = self.predict(x[i, :].reshape(1, x.shape[1]))
            prob_list.append(prob)
        return prob_list

    def prob_0(self, x):
        """
        :param x: shape(1, feature_n)
        :return: float 为0的概率值
        """
        x = self.scaler.transform(x)
        item1 = np.dot(x[0, :], self.w[0, :])
        y = self.w0 + item1 + self.cross_item(x)
        return self.sigmoid(y)

    def cross_item(self, x):
        """
        :param x: shape(1, feature_n)
        :return: float 交叉项的值
        """
        total = 0
        for j1 in range(self.feature_n):
            for j2 in range(j1+1, self.feature_n):
                j1_field = self.feature_filed_dict[j1]
                j2_field = self.feature_filed_dict[j2]
                v_j1_fj2 = self.v[j1, j2_field, :]
                v_j2_fj1 = self.v[j2, j1_field, :]
                total += np.dot(v_j1_fj2, v_j2_fj1) * x[0, j1] * x[0, j2]
        return total

if __name__ == "__main__":
    import pandas as pd
    x, y = load_data()
    print(pd.Series(y).value_counts())
    feature_field_dict = {}
    for i in range(x.shape[1]):
        if i <=10:
            feature_field_dict[i] = 0
        elif i <= 20:
            feature_field_dict[i] = 1
        else:
            feature_field_dict[i] = 2
    for i in range(10):
        ffm = FFM(feature_n=x.shape[1], hidden_n=3, field_n=3, feature_field_dict=feature_field_dict)
        ffm.fit(x, y, eta=0.9, alpha=0.0001, iter_n=1)
        prob = ffm.predict(x[0, :].reshape(1, x.shape[1]))
        prob_list = ffm.predict_all(x)
        auc = roc_auc_score(y, prob_list)
        print(auc)
