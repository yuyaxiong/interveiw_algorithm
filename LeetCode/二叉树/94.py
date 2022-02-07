# Definition for a binary tree node.
# 94. 二叉树的中序遍历 
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        self.recur_mid(root, result)
        return result

    def recur_mid(self, root, result):
        if root is None:
            return 
        self.recur_mid(root.left, result)
        result.append(root.val)
        self.recur_mid(root.right, result)


# 非递归版本
class Solution1:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        result = []
        tmp = [root]
        while len(tmp) > 0:
            node = tmp.pop(0)
            new_tmp = []
            if node.left is not None:
                new_tmp.append(node.left)
                node.left = None
                new_tmp.append(node)
                tmp = new_tmp + tmp
                continue
            else:
                result.append(node.val)
            if node.right is not None:
                new_tmp.append(node.right)
                tmp = new_tmp + tmp
                continue
        return result


def testCase():
    tn1 = TreeNode(val=1)
    tn2 = TreeNode(val=2)
    tn3 = TreeNode(val=3)
    tn1.right = tn2
    tn2.left = tn3
    
    sol = Solution1()
    ret = sol.inorderTraversal(tn1)
    print(ret)


if __name__ == "__main__":
    testCase()