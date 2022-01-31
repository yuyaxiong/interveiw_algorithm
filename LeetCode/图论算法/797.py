# 797. 所有可能的路径

from typing import List


class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        pre_list = [0]
        path_list = []
        self.getPath(graph[0], graph, path_list, pre_list)
        return path_list

    def getPath(self, n_list, graph, path_list, pre_list):
        for i in n_list:
            if i == len(graph)-1:
                tmp = pre_list[::]
                tmp.append(i)
                path_list.append(tmp)
            else:
                self.getPath(graph[i], graph, path_list, pre_list + [i])
            
