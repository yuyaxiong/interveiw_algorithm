# 210. 课程表 II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        self.postorder = []
        self.has_clcle = False
        self.visited, self.on_path = [False for _ in range(numCourses+1)], [False for _ in range(numCourses+1)]
        graph = self.buildGraph(numCourses, prerequisites)
        # 遍历图
        for i in range(numCourses):
            self.traverse(graph, i)
        # 有环就直接返回
        if self.has_clcle:
            return []
        self.postorder = self.postorder[::-1]
        return self.postorder

    def traverse(self, graph, s):
        if self.on_path[s]:
            self.has_clcle = True
        if self.visited[s] or self.has_clcle:
            return 
        # 前序遍历
        self.on_path[s] = True
        self.visited[s] = True
        for t in graph[s]:
            self.traverse(graph, t)
        self.on_path[s] = False
        # 后序遍历 这里整个图遍历的时候不会有问题吗？
        self.postorder.append(s)
        return 


    def buildGraph(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses+1)]
        for edge in prerequisites:
            # 单向
            froms, tos = edge[1], edge[0]
            graph[froms].append(tos)
        return graph
