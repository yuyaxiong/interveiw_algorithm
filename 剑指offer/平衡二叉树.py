"""
输入一颗二叉树的根节，判断该树是不是平衡二叉树。如果某二叉树中
任意节点的左右子树的深度相差不超过1，那么它就是一颗平衡二叉树。
例如：下面为平衡二叉树。
        1
    2      3
  4   5       6
    7
"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None


class Solution(object):
    def is_balance(self, pRoot):
        if pRoot is None:
            return True
        left = self.depth(pRoot.left)
        right = self.depth(pRoot.right)
        if abs(left - right) > 1:
            return False
        return self.is_balance(pRoot.left) and self.is_balance(pRoot.right)

    def depth(self, pRoot):
        if pRoot is None:
            return 0
        left = self.depth(pRoot.left)
        right = self.depth(pRoot.right)
        return max(left, right) + 1

class Solution1(object):
    def is_balance(self, pRoot):
        if pRoot is None:
            return 0, True

        left, flag_left = self.is_balance(pRoot.left)
        right, flag_right = self.is_balance(pRoot.right)

        if flag_left and flag_right:
            if abs(left - right) <= 1:
                return max(left, right) + 1, True

        return max(left, right)+1, False

if __name__ == '__main__':
    pRoot = BinaryTree()
    pRoot.value = 5
    # pRoot.left = BinaryTree()
    # pRoot.left.value = 3
    # pl = pRoot.left
    pRoot.right = BinaryTree()
    pRoot.right.value = 7
    pr = pRoot.right
    # pl.left = BinaryTree()
    # pl.right = BinaryTree()
    pr.left = BinaryTree()
    pr.right = BinaryTree()
    # pl.left.value = 2
    # pl.right.value = 4
    # pr.left.value =
    pr.right.value = 8

    s = Solution()
    print(s.is_balance(pRoot))







