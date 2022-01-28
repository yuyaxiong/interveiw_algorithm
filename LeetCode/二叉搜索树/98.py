# Definition for a binary tree node.
# 98. 验证二叉搜索树
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if root is None:
            return True
        elif root.left is None and root.right is not None:
            if root.val < self.getLeftNode(root.right).val:
                return self.isValidBST(root.right)
            else:
                return False
        elif root.left is not None and root.right is None:
            if self.getRigthNode(root.left).val < root.val:
                return self.isValidBST(root.left)
            else:
                return False
        elif root.left is not None and root.right is not None:
            if root.val > self.getRigthNode(root.left).val and root.val < self.getLeftNode(root.right).val:
                return self.isValidBST(root.left) and self.isValidBST(root.right)
            else:
                return False
        else:
            return True

    def getLeftNode(self, node):
        if node.left is None:
            return node
        return self.getLeftNode(node.left) 

    def getRigthNode(self, node):
        if node.right is None:
            return node
        return self.getRigthNode(node.right)


def testCase():
    tn1 = TreeNode(val=5)
    tn2 = TreeNode(val=1)
    tn3 = TreeNode(val=4)
    tn4 = TreeNode(val=3)
    tn5 = TreeNode(val=6)
    tn1.left = tn2
    tn1.right = tn3
    tn3.left = tn4
    tn3.right = tn5
    sol = Solution()
    ret = sol.isValidBST(tn1)
    print(ret)
    # print(sol.getLeftNode(tn1).val)
    # print(sol.getRigthNode(tn1).val)
    print(sol.getRigthNode(tn1.left).val)
    print(sol.getLeftNode(tn1.right).val)
    



if __name__ == "__main__":
    testCase()



