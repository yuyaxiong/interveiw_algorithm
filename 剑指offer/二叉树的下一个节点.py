"""
给定一颗二叉树和其中一个节点，如何找出中序遍历序列的下一个节点？
树中的节点除了有连个分别指向左，右子节点的指针，还有一个指向
父节点的指针。
        a
    b       c
  d   e   f    g
     h  i


"""

class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None
        self.father = None

class Solution(object):
    def binary_tree_next(self, node):
        root = node.father
        if root.left == node:
            return root
        else:
            if node.right is not None:
                return self.find_left(root.right)
            else:
                return self.find_father_left(root)

    def find_left(self, root):
        if root.left is None:
            return root
        return self.find_left(root.left)

    def find_father_left(self, root):
        father = root.father
        if father.left == root:
            return root
        return self.find_father_left(root.father)


if __name__ == '__main__':
    pass
