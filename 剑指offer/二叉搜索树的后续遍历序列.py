"""
输入一个整数数组，判断该数组是否是某二叉搜索树的后续遍历结果。如果是则返回True，否则返回False。
假设输入的数组的任意两个数字都互不相同。例如，输入数组（5，7，6，9，11，10，8），则返回True，
因为这个整数序列是图4.9二叉搜索树的后续遍历结果。如果输入的数组是（7，4，6，5），则由于没有哪
颗二叉搜索树的后续遍历结果是这个序列，因此返回False。
        8
     6      10
   5    7 9     11
"""

class Solution(object):
    def verify_sequence_of_BST(self, nList):
        if len(nList) == 0:
            return True

        root = nList[-1]
        mid = 0
        for idx, n in enumerate(nList):
            if n >= root:
                mid = idx
                break
        left, right = nList[:mid], nList[mid:-1]
        for n in right:
            if n < root:
                return False
        return self.verify_sequence_of_BST(left) and self.verify_sequence_of_BST(right)


if __name__ == '__main__':
    s = Solution()
    # nList = [5, 7, 6,9, 11, 10, 8]
    nList = [7, 4, 6, 5]
    print(s.verify_sequence_of_BST(nList))








