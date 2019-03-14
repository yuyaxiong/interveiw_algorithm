"""
输入一个二叉树的Root节点和两个树节点，求它们的最低公共父节点。

        A
    B       C
  D    E
F   G H  J
"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None


class Solution(object):
    # 判断左右子树中的p，q节点
    def common_father(self, root, p, q):
        if not self.in_substree(root, p) or not self.in_substree(root, q):
            return -1
        return self.common_father_help(root, p, q)

    def common_father_help(self, root, p, q):
        if root is None:
            return
        if self.in_substree(root.left, p) and self.in_substree(root.left, q):
            return self.common_father_help(root.left, p, q)
        elif self.in_substree(root.right, p) and self.in_substree(root.right, q):
            return self.common_father_help(root.right, p, q)
        else:
            return root

    def in_substree(self, root, p):
        if root is None:
            return False
        if root.value == p:
            return True
        return self.in_substree(root.left, p) or self.in_substree(root.right, p)

class Solution1(object):
    # 找出路径并判断
    def common_father(self, root, p, q):
        pList = self.get_path(root, p)
        qList = self.get_path(root, q)
        idx = 0
        while pList[idx] == qList[idx]:
           idx += 1
        return pList[idx-1]

    def get_path(self, root, p):
        path = []; result = []
        result = self.get_path_help(root, p, path, result)
        return result

    def get_path_help(self, root, p,  path, result):
        if root is None:
            return result

        tmp = path[::]
        tmp.append(root.value)
        if root.value == p:
            return tmp

        if len(result) == 0:
            result = self.get_path_help(root.left, p, tmp, result)
        if len(result) == 0:
            result = self.get_path_help(root.right, p, tmp, result)
        return result


class BinaryTreePath(object):
    def get_path(self, pRoot):
        result = self.get_path_help(pRoot, [], [])
        return result

    def get_path_help(self, pRoot, path, result):
        if pRoot.left is None and pRoot.right is None:
            tmp1 = path[::]
            tmp1.append(pRoot.value)
            result.append(tmp1)
            return result
        tmp = path[::]
        tmp.append(pRoot.value)
        self.get_path_help(pRoot.left, tmp, result)
        self.get_path_help(pRoot.right, tmp, result)
        return result

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
    pr.left.value = 6
    pr.right.value = 8

    # s = Solution()
    # print(s.common_father(pRoot, 2, 4).value)
    s = Solution1()
    print(s.common_father(pRoot, 8, 4))
    print(s.get_path(pRoot, 4))
    # bt = BinaryTreePath()
    # print(bt.get_path(pRoot))
