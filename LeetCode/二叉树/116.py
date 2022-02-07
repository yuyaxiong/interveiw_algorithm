"""
# Definition for a Node.

"""
# 116. 填充每个节点的下一个右侧节点指针
from typing import Optional

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution:
    def connect(self, root: Optional[Node]) -> Optional[Node]:
        if root is None:
            return None
        node_list = [root]
        while len(node_list) > 0:
            nodes = []
            for i in range(len(node_list)):
                if i + 1 < len(node_list):
                    next_node = node_list[i+1]
                else:
                    next_node = None
                node_list[i].next = next_node
                if node_list[i].left is not None:
                    nodes.append(node_list[i].left)
                if node_list[i].right is not None:
                    nodes.append(node_list[i].right)
            node_list = nodes
        return root
