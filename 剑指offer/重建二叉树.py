"""
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历
和中序遍历的结果中都不含重复的数字。例如，输入前序遍历序列[1,2,4,7,3,5,6,8]
和中序遍历序列[4,7,2,1,5,3,8,6]，则重建如下二叉树并输出它的头节点。
        1
    2      3
  4      5    6
     1      8

"""


class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

class Solution(object):
    def binary_tree_constrcut(self, preorder, inorder):
        if len(preorder) == 0:
            return None
        value = preorder[0]
        idx = self.find_idx(inorder, value)
        left_inorder = inorder[:idx]
        right_inorder = inorder[idx+1:]
        left_preorder = preorder[1:1+len(left_inorder)]
        right_preorder = preorder[1+len(left_inorder):]
        TreeNode=BinaryTree()
        TreeNode.value = value
        TreeNode.left = self.binary_tree_constrcut(left_preorder, left_inorder)
        TreeNode.right = self.binary_tree_constrcut(right_preorder, right_inorder)
        return TreeNode

    def find_idx(self, nList, num):
        for idx, n in enumerate(nList):
            if n == num:
                return idx


if __name__ == "__main__":
    s = Solution()
    preorder = [1, 2, 4, 7, 3, 5, 6, 8]
    inorder = [4, 7, 2, 1, 5, 3, 8, 6]
    root = s.binary_tree_constrcut(preorder=preorder, inorder=inorder)
    print(root.value)