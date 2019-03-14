"""
输入两棵二叉树A和B，判断B是不是A的子结构。
"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

class Solution(object):
    def hasSubTree(self, pRoot1, pRoot2):
        result = False
        if pRoot1.value == pRoot2.value:
            # 两个递归，两个不同的实现逻辑
            result = self.DoesTreeHaveTree2(pRoot1, pRoot2)
        if result is False:
            result = self.hasSubTree(pRoot1.left, pRoot2)
        if result is False:
            result = self.hasSubTree(pRoot1.right, pRoot2)
        return result

    def DoesTreeHaveTree2(self, pRoot1, pRoot2):
        if pRoot2 is None:
            return True
        if pRoot1 is None:
            return False
        if pRoot1.value != pRoot2.value:
            return False
        return self.DoesTreeHaveTree2(pRoot1.left, pRoot2.left) and self.DoesTreeHaveTree2(pRoot1.right, pRoot2.right)





