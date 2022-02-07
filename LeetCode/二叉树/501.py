# Definition for a binary tree node.
# 501. 二叉搜索树中的众数
from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def __init__(self):
        self.pre_node = None
        self.cur_count = 0
        self.max_count = 0
        self.mode_list = []

    def findMode(self, root: TreeNode) -> List[int]:
        self.traverse(root)
        return self.mode_list

    def traverse(self, root):
        if root is None:
            return
        self.traverse(root.left)
        # 开头
        if self.pre_node is None:
            self.cur_count = 1
            self.max_count = 1
            self.mode_list.append(root.val)
        else:
            if root.val == self.pre_node.val:
                self.cur_count += 1
                if self.cur_count == self.max_count:
                    self.mode_list.append(root.val)
                elif self.cur_count > self.max_count:
                    self.mode_list.clear()
                    self.max_count = self.cur_count
                    self.mode_list.append(root.val)
            elif root.val != self.pre_node.val:
                self.cur_count = 1
                if self.cur_count == self.max_count:
                    self.mode_list.append(root.val)
        self.pre_node = root
        self.traverse(root.right)





