# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 236. 二叉树的最近公共祖先
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return None
        # 这里处理p=5 q=4的情况
        if root.val == p.val or root.val == q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left is not None and right is not None:
            return root 
        if left is None and right is None:
            return None
        if left is None:
            return right 
        else:
            return left

# 会超时 因为递归在反复计算
class Solution1:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return root
        if (root.val == p.val and self.is_exists(root, q)) or (root.val == q.val and self.is_exists(root, p)):
            return root
        left_p, right_p = self.is_exists(root.left, p), self.is_exists(root.righ, p)
        left_q, right_q = self.is_exists(root.left, q), self.is_exists(root.right, q)

        if (left_p and right_q) or (left_q and right_p):
            return root
        elif left_p and left_q:
            return self.lowestCommonAncestor(root.left, p, q)
        elif right_p and right_q:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return None

    def is_exists(self, root, node):
        if root is None:
            return False 
        if root.val == node.val:
            return True
        return self.is_exists(root.left, node) or self.is_exists(root.right, node)









def testCase():
    tn1 = TreeNode(3)
    tn2 = TreeNode(5)
    tn3 = TreeNode(1)
    tn4 = TreeNode(6)
    tn5 = TreeNode(2)
    tn6 = TreeNode(0)
    tn7 = TreeNode(8)
    tn8 = TreeNode(7)
    tn9 = TreeNode(4)
    tn1.left = tn2
    tn1.right = tn3
    
    tn2.left = tn4
    tn2.right = tn5

    tn3.left = tn6
    tn3.right = tn7

    tn5.left = tn8
    tn5.right = tn9

    q = TreeNode(5)
    p = TreeNode(4)
    sol = Solution1()
    ret = sol.lowestCommonAncestor(tn1, q, p)

    print(ret)
    print(ret.val)
    # ret = sol.is_exists(tn1, p)
    # print(ret)


def testCase1():
    tn1 = TreeNode(1)
    tn2 = TreeNode(2)
    tn3 = TreeNode(3)
    tn1.left = tn2
    tn1.right = tn3

    p = TreeNode(3)
    q = TreeNode(2)
    sol = Solution()
    ret = sol.lowestCommonAncestor(tn1, p, q)
    print(ret.val)


if __name__ == "__main__":
    testCase()
    testCase1()

    
