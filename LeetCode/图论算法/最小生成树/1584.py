# 1584.连接所有点的最小费用
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

# 1584.连接所有点的最小费用
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        edges = []
        for i in  range(n):
            for j in range(i+1, n):
                xi, yi = points[i][0], points[i][1]
                xj, yj = points[j][0], points[j][1]
                edges.append([i, j, abs(xi - xj) + abs(yi-yj)])
        edges = sorted(edges, key=lambda x: x[2])
        mst = 0.0
        uf = UF(n)
        for edge in edges:
            u, v, weight = edge[0], edge[1], edge[2]
            if uf.connected(u, v):
                continue
            mst += weight
            uf.union(u, v)

        return mst

    
        