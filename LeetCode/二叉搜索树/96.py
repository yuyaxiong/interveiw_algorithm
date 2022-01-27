
# 96. 不同的二叉搜索树
class Solution:
    def numTrees(self, n: int) -> int:
        if n == 0:
            return 0
        self.val_list = [[0 for _ in range(n+2)] for _ in range(n+2)]
        return self.countTree(1, n+1)

    def countTree(self, left, right):
        if left >= right:
            return 1
        res = 0
        if self.val_list[left][right] != 0:
            return self.val_list[left][right]
        for i in range(left, right):
            left_count = self.countTree(left, i)
            right_count = self.countTree(i+1, right)
            res += left_count * right_count
        self.val_list[left][right] = res
        return res

