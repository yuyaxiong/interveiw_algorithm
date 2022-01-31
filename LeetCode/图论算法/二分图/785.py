

from typing import List

# 判断二分图
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        self.ok = True
        self.color = [False for _ in range(n)]
        self.visited = [False for _ in range(n)]
        for v in range(n):
            if self.visited[v] is False:
                self.traverse(graph, v)
        return self.ok

    def traverse(self, graph, v):
        if self.ok is False:
            return 
        self.visited[v] = True
        for w in graph[v]:
            if self.visited[w] is False:
                self.color[w]  = not self.color[v]
                self.traverse(graph, w)
            else:
                if self.color[w] == self.color[v]:
                    self.ok = False
                

        
                