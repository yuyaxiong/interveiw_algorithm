"""
# Definition for a Node.
"""

from typing import List


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if root is None:
            return []
        self.result = []
        self.postorderHelp(root)
        return self.result

    def postorderHelp(self, root):
        if root.children is None:
            self.result.append(root.val)
            return 
        for node in root.children:
            self.postorderHelp(node)
        self.result.append(root.val)


class Solution1:
    def postorder(self, root: 'Node') -> List[int]:
        if root is None:
            return []
        result = []
        node_list = [root]
        while len(node_list) > 0:
            node = node_list.pop(0)
            node_childen = node.children
            node.children = None
            if node_childen is not None:
                node_childen.append(node)
                node_childen.extend(node_list)
                node_list = node_childen
            else:
                result.append(node.val)
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
    ret = sol.postorder(tn1)
    print(ret)

    sol = Solution1()
    ret1 = sol.postorder(tn1)
    print(ret1)

if __name__ == "__main__":
    testCase()
