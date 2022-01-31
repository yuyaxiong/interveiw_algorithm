# Definition for a binary tree node.
from typing import Optional

# 二叉树最大深度
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        return self.getDepth(root)

    def getDepth(self, root):
        if root is None:
            return 0
        return max(self.getDepth(root.left), self.getDepth(root.right)) + 1

def testCase():
    tn1 = TreeNode(val=1)
    tn2 = TreeNode(val=9)
    tn3 = TreeNode(val=20)
    tn4 = TreeNode(val=15)
    tn5 = TreeNode(val=7)
    tn1.left = tn2 
    tn1.right = tn3
    tn3.left = tn4
    tn3.right = tn5
    sol = Solution()
    ret = sol.maxDepth(tn1)
    print(ret)




if __name__ == "__main__":
    testCase()
