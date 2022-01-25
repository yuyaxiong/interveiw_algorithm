# Definition for a binary tree node.
from typing import List, Optional

#后序遍历
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 递归版本
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        result = []
        self.postOrder(root, result)
        return result

    def postOrder(self, root, result):
        if root is None:
            return []
        self.postOrder(root.left, result)
        self.postOrder(root.right, result)
        result.append(root.val)

# 非递归版本
class Solution1:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        result = []
        node_list = [root]
        while len(node_list) > 0:
            print(node_list)
            node = node_list.pop(0)
            if node.left is None and node.right is None:
                result.append(node.val)
            else:
                tmp = []
                if node.left is not None:
                    tmp.append(node.left)
                if node.right is not None:
                    tmp.append(node.right)
                node.left, node.right = None, None
                tmp.append(node)
                node_list = tmp + node_list
        return  result


def testCase():
    tn1 = TreeNode(val=1)
    tn2 = TreeNode(val=2)
    tn3 = TreeNode(val=3)
    tn1.right = tn2
    tn2.left = tn3
    sol = Solution1()
    res = sol.postorderTraversal(tn1)
    print(res)

if __name__ == "__main__":
    testCase()

