# 1514. 概率最大的路径

from typing import List
import sys

class State:
    def __init__(self,cur_id, cur_prob):
        self.cur_id = cur_id
        self.cur_prob = cur_prob

class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        graph = self.buildGraph(n, edges, succProb)
        print(graph)
        probTo = [0 for _ in range(n)]
        pq = []
        pq.append(State(start, 1))
        probTo[start] = 1
        # 广度优先遍历
        while len(pq) > 0:
            cur_state = pq.pop(0)
            cur_id = cur_state.cur_id
            cur_prob = cur_state.cur_prob

            if probTo[cur_id] > cur_prob:
                continue
            for neighbor in graph[cur_id]:
                next_id, prob = neighbor[0], neighbor[1]
                next_prob = cur_prob * prob
                if probTo[next_id] < next_prob:
                    probTo[next_id] = next_prob 
                    pq.append(State(next_id, next_prob))
        return probTo[end]
            

    def buildGraph(self, n: int, edges:List[List[int]], succProb: List[float]) -> List[List[int]]:
        graph = [[] for _ in range(n)]
        for edge, prob in zip(edges, succProb):
            u, v = edge[0], edge[1]
            graph[u].append([v, prob])
            graph[v].append([u, prob])
        return graph
            


def testCase():
    n = 3
    edges = [[0,1],[1,2],[0,2]]
    succProb = [0.5,0.5,0.2]
    start = 0 
    end = 2
    sol = Solution()
    ret = sol.maxProbability(n , edges, succProb, start, end)
    print(ret)

def testCase1():
    n = 5
    edges = [[1,4],[2,4],[0,4],[0,3],[0,2],[2,3]]
    succProb = [0.37,0.17,0.93,0.23,0.39,0.04]
    start = 3
    end = 4
    sol = Solution()
    ret = sol.maxProbability(n , edges, succProb, start, end)
    print(ret)


if __name__ == "__main__":
    testCase()
    testCase1()
    


        