import pandas as pd
import math

def calc_entropy(y):
    vc = y.value_counts()/len(y)
    entropy = 0
    for p in vc.tolist():
        entropy += - p * math.log(p, 2)
    return entropy

def feature_entropy(x, y):
    entropy = 0
    for bins in x.value_counts().tolist():
        subset_y = y[x == bins]
        entropy += calc_entropy(subset_y)
    return entropy

def find_best_feature(x, y):
    base_entropy = calc_entropy(y)
    best_feat = x.columns.tolist()[0]
    best_gain_info = base_entropy - feature_entropy(x[best_feat], y)
    for col in x.columns.tolist():
        gain_info = base_entropy - feature_entropy(x[best_feat], y)
        if best_gain_info < gain_info:
            best_gain_info = gain_info
            best_feat = col
    return best_feat

def max_class(y):
    vc = y.value_counts()
    return vc[vc == vc.max()].index[0]

def dt_fit(x, y):
    feat_list = x.columns.tolist()
    if len(feat_list) == 0:
        return max_class(y)
    best_feature = find_best_feature(x,y)
    feat_list.remove(best_feature)
    tree = {best_feature:{}}
    for bins in x[best_feature].value_counts().index.tolist():
        subset_x = x[x[best_feature] == bins][feat_list]
        subset_y = y[x[best_feature] == bins]
        tree[best_feature][bins] = dt_fit(subset_x, subset_y)
    return tree

def dt_predict(tree, x):
    root = list(tree.keys())[0]
    val_dict = tree[root]
    subtree = val_dict[x[root]]
    sub_x = x.drop(root)
    if isinstance(subtree, dict):
        return dt_predict(subtree, sub_x)
    else:
        return subtree

if __name__ == '__main__':
    x = [[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]]
    y = pd.Series(['yes', 'yes', 'no', 'no', 'no'])
    x = pd.DataFrame(x, columns=['feat_1', 'feat_2'])
    my_tree = dt_fit(x, y)
    print(my_tree)
    for idx in x.index.tolist():
        print(dt_predict(my_tree, x.loc[idx, :]))