# Definition for a binary tree node.
from typing import List

# 106. 从中序与后序遍历序列构造二叉树
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if len(postorder) == 0:
            return None
        val = postorder.pop()
        node = TreeNode(val)
        inorder_idx = inorder.index(val)
        left_inorder = inorder[:inorder_idx]
        right_inorder = inorder[inorder_idx+1:]
        left_postorder, right_postorder = self.getPostOrderLeftRight(postorder, left_inorder)
        node.left = self.buildTree(left_inorder, left_postorder)
        node.right = self.buildTree(right_inorder, right_postorder)
        return node


    def getPostOrderLeftRight(self, postorder, left_inorder):
        left_inorder_dict = {i:1 for i in left_inorder}
        left_postorder, right_postorder = [] , []
        for o in postorder:
            if o in left_inorder_dict:
                left_postorder.append(o)
            else:
                right_postorder.append(o)
        return left_postorder, right_postorder

            
        
        
