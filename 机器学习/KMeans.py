import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def load_data():
    data = load_breast_cancer()
    return pd.DataFrame(data.data), pd.DataFrame(data.target)

class KMeansR(object):
    def __init__(self):
        self.n_cluster = None
        self.num_iter = None
        self.feature_list = None

    def fit(self, data):
        self.cluster = [data.loc[idx, :] for idx in data.sample(2).index.tolist()]
        for i in range(self.num_iter):
            # 计算样本归属于那个簇
            data['cluster'] = data[self.feature_list].apply(lambda x: self.min_distance(x, self.cluster), axis=1)
            print('iteration: %s' % i)
            # 重新计算簇中心
            df = data.groupby('cluster').mean().sort_index()
            self.cluster = [df.loc[idx, :] for idx in df.index.tolist()]
        return self.cluster

    def predict(self, data):
        data['cluster'] = data[self.feature_list].apply(lambda x: self.min_distance(x, self.cluster), axis=1)
        return data['cluster']

    def min_distance(self, x, cluster):
        distance_list = []
        for idx, n in enumerate(cluster):
            distance = np.sum((x.values - n)**2) ** (1/2)
            distance_list.append(distance)
        return distance_list.index(min(distance_list))

if __name__ == '__main__':
    #由于是聚类，所以无法把预测的0，1标签与target中的0，1标签对应上，因此会出现AUC小于0.5的情况，对上了其实就是1-0.19=0.8

    feature, target = load_data()
    feature_list = feature.columns.tolist()
    cluster = KMeansR()
    cluster.n_cluster = 2
    cluster.num_iter = 10
    cluster.feature_list = feature_list
    cluster.fit(feature)
    feature['cluster2'] = cluster.predict(feature)
    print(roc_auc_score(target, feature['cluster2']))
    cluster1 = KMeans(n_clusters=2, max_iter=100)
    cluster1.fit(feature)
    feature['cluster1'] = cluster1.predict(feature)
    print(roc_auc_score(target, feature['cluster1']))
    lr_model = LogisticRegression()
    lr_model.fit(feature, target)
    feature['prob'] = lr_model.predict(feature)
    print(roc_auc_score(target, feature['prob']))





