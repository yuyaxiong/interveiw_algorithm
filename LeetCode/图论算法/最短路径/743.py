# 743. 网络延迟时间
from typing import List
import sys

class State:
    def __init__(self, id, distFromStart):
        self.id = id 
        self.dist_from_start = distFromStart

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # 节点从1开始，所以需要一个n+1的领接表
        graph = [[] for _ in range(n+1)]
        # 构造图
        for edge in times:
            froms, tos, weight = edge[0], edge[1], edge[2]
            graph[froms].append([tos, weight])
        print(graph)
        distTo = self.dijkstra(k, graph)
        print(distTo)
        res = 0
        for dist in distTo[1:]:
            if dist == sys.maxsize:
                return -1
            res = max(res, dist)
        return res

    def dijkstra(self, start:int, graph:List[List[List[int]]]):
        # 用于记录节点的最小距离
        distTo = [sys.maxsize for _ in range(len(graph))]
        distTo[start] = 0

        pq = []
        # 从start节点开始进行BFS
        pq.append(State(start, 0))

        while len(pq) > 0:
            cur_state = pq.pop(0)
            cur_node_id = cur_state.id
            cur_dist_from_start = cur_state.dist_from_start

            if cur_dist_from_start > distTo[cur_node_id]:
                continue
            # 将cur_node 的相邻节点装入队列
            for neighbor in graph[cur_node_id]:
                next_node_id = neighbor[0]
                dist_to_next_node = distTo[cur_node_id] + neighbor[1]
                if distTo[next_node_id] > dist_to_next_node:
                    distTo[next_node_id] = dist_to_next_node
                    pq.append(State(next_node_id, dist_to_next_node))
        return distTo

def testCase():
    times = [[2,1,1],[2,3,1],[3,4,1]]
    n = 4
    k = 2
    sol = Solution()
    ret = sol.networkDelayTime(times, n, k)
    print(ret)

if __name__ == "__main__":
    testCase()
