# 886. 可能的二分法

class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        graph = self.buildGraph(n, dislikes)
            
        self.visited = [False for _ in range(len(graph))]
        self.color = [False for _ in range(len(graph))]
        self.ok = True
        for v in range(1, n+1):
            if self.visited[v] is False:
                self.trans(graph, v)
        return self.ok

    def buildGraph(self, n, dislikes):
        graph = [[] for _ in range(n+1)]
        for edge in dislikes:
            v, w = edge[1], edge[0]
            if graph[v] is None:
                graph[v] = []
            graph[v].append(w)
            if graph[w] is None:
                graph[w] = []
            graph[w].append(v)
        return graph

    def trans(self, graph, v):
        if self.ok is False:
            return 
        self.visited[v] = True
        for w in graph[v]:
            if self.visited[w] is False:
                self.color[w] = not self.color[v]
                self.trans(graph, w)
            else:
                if self.color[w] == self.color[v]:
                    self.ok = False

        
