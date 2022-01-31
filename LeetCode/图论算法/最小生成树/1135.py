from typing import List


class UF():
    def __init__(self, n):
        self.count = n
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return None 
        # 小树挂大树下面
        if self.size[rootP] > self.size[rootQ]:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count -= 1 

    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def count(self):
        return self.count

# 1135.最低成本连通所有城市
class Solution:
    def minimumCost(n:int, connections: List[List[int]]):
        uf = UF(n+1)
        # 最小生成树，按照weight升序排序
        connections = sorted(connections, key=lambda x: x[2])
        mst = 0.0
        for edge in connections:
            u, v, weight = edge[0], edge[1], edge[2]
            # 成环就跳过
            if uf.connected(u, v):
                continue
            mst += weight
            uf.union(u, v)
        # 保证所有节点连通，因为0没有被使用，额外占有一个，因此总共是2
        return mst if uf.count() == 2 else -1



