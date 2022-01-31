# 207. 课程表
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 递归的点，防止递归的loop
        self.on_path = [False for _ in range(numCourses+1)]
        # 遍历的点，防止遍历的loop
        self.visited = [False for _ in range(numCourses+1)]
        self.has_clcle = False
        graph = self.buildGraph(numCourses, prerequisites)
        for i in range(numCourses):
            self.traverse(graph, i)
        return not self.has_clcle

    def buildGraph(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        for edge in prerequisites:
            froms = edge[1]
            tos = edge[0]
            graph[froms].append(tos)
        return graph

    def traverse(self, graph, i):
        if self.on_path[i] is True:
            self.has_clcle = True
        if self.visited[i] or self.has_clcle:
            return 
        self.visited[i] = True
        self.on_path[i] = True
        for t in graph[i]:
            self.traverse(graph, t)
        self.on_path[i] = False


            

