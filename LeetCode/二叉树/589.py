"""
# Definition for a Node.
"""

from typing import List

# 589. N 叉树的前序遍历
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if root is None:
            return []
        self.result = []
        self.preoderHelp(root)
        return self.result

    def preoderHelp(self, root):
        if root is None:
            return 
        self.result.append(root.val)
        if root.children is not None:
            for node in root.children:
                self.preoderHelp(node)

# 非递归遍历
class Solution1:
    def preorder(self, root: 'Node') -> List[int]:
        if root is None:
            return []
        node_list = [root]
        result = []
        while len(node_list) > 0:
            node = node_list.pop(0)
            result.append(node.val)
            if node.children is not None:
                node_list = node.children + node_list
        return result


def testCase():
    tn1 = Node(val=1)
    tn2 = Node(val=3)
    tn3 = Node(val=2)
    tn4 = Node(val=4)

    tn5 = Node(val=5)
    tn6 = Node(val=6)
    tn1.children = [tn2, tn3, tn4]
    tn2.children = [tn5, tn6]
    
    sol = Solution()
    ret = sol.preorder(tn1)
    print(ret)

    sol = Solution1()
    ret1 = sol.preorder(tn1)
    print(ret1)


if __name__ == "__main__":
    testCase()

    
