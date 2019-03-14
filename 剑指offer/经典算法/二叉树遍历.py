"""
二叉树遍历,非递归方式遍历
"""
class BinaryTree(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None


class Solution(object):

    # 前序遍历
    def preorder_print_node(self, pRoot):
        heap = [pRoot]
        result = []
        while len(heap) != 0:
            root = heap.pop()
            if root.value is not None:
                result.append(root.value)
            if root.right is not None:
                heap.append(root.right)
            if root.left is not None:
                heap.append(root.left)
        return result

    # 中序遍历
    def inorder_print_node(self, pRoot):
        if pRoot is None:
            return None
        else:
            node_stack = []
            result = []
            node = pRoot
            while node is not None or len(node_stack) != 0:
                if node is None:
                    node = node_stack.pop()
                    result.append(node.value)
                    node = node.right
                    continue
                while node.left is not None:
                    node_stack.append(node)
                    node = node.left
                if node.value is not None:
                    result.append(node.value)
                node = node.right
        return result

    # 后序遍历
    def post_print_node(self, pRoot):
        if pRoot is None:
            return None
        else:
            node_stack = []
            result = []
            node = pRoot
            while node is not None or len(node_stack) != 0:
                if node is None:
                    # 与中序遍历的唯一区别,没访问过就先访问右子树，并将状态置为True。
                    # 如果访问过了就直接append，并将该节点pop。
                    visited = node_stack[-1]['visited']
                    if visited:
                        result.append(node_stack[-1]['node'].value)
                        node_stack.pop()
                    else:
                        node_stack[-1]['visited'] = True
                        node = node_stack[-1]['node']
                        node = node.right
                    continue
                while node.left is not None:
                    node_stack.append({'node': node, 'visited': False})
                    node = node.left
                if node.value is not None:
                    result.append(node.value)
                node = node.right
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
    # pr.left.value =
    pr.right.value = 8

    s = Solution()
    print(s.preorder_print_node(pRoot))
    print(s.inorder_print_node(pRoot))
    print(s.post_print_node(pRoot))
