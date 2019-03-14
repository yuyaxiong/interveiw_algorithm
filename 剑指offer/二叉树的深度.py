"""
输入一颗二叉树的根节点，求该树的深度。从根节点到叶节点一次经过的节点（含根，叶节点）形成树的一条路径，
最长路径的长度为树的深度。
      5
    3   7
  2       8
    1
"""

class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None


class Solution(object):
    def tree_depth(self, pRoot):
        depth = 0
        current = 0
        return self.depth_help(pRoot, depth, current)


    def depth_help(self, pRoot, depth, current):
        if pRoot is None:
            return depth
        current += 1
        depth = max(depth, current)
        depth = self.depth_help(pRoot.left, depth, current)
        depth = self.depth_help(pRoot.right, depth, current)
        return depth

class Solution1(object):
    def tree_depth(self, pRoot):
        if pRoot is None:
            return 0
        left = self.tree_depth(pRoot.left)
        right = self.tree_depth(pRoot.right)
        return max(left, right) + 1

if __name__ == '__main__':
    pRoot = BinaryTree()
    pRoot.value = 5
    pRoot.left = BinaryTree()
    pRoot.left.value = 3
    pl = pRoot.left
    pRoot.right = BinaryTree()
    pRoot.right.value = 7
    pr = pRoot.right
    pl.left = BinaryTree()
    pl.right = BinaryTree()
    pr.left = BinaryTree()
    pr.right = BinaryTree()
    pl.left.value = 2
    pl.right.value = 4
    # pr.left.value =
    pr.right.value = 8

    s = Solution1()
    print(s.tree_depth(pRoot))