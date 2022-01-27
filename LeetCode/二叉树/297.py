# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
# 297. 二叉树的序列化与反序列化
class Codec:
    def __init__(self):
        self.sep=','
        self.null="#"


    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        result_list = []
        self.serHelp(root, result_list)
        return self.sep.join(result_list)

    def serHelp(self, root, result_list):
        if root is None:
            result_list.append(self.null)
            return 
        result_list.append(str(root.val))
        self.serHelp(root.left, result_list)
        self.serHelp(root.right, result_list)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        data_list = data.split(self.sep)
        return self.desHelp(data_list)

    def desHelp(self, data_list):
        if len(data_list) == 0:
            return None
        first = data_list.pop(0)
        if first == self.null:
            return None
        node = TreeNode(int(first))
        node.left = self.desHelp(data_list)
        node.right = self.desHelp(data_list)
        return node



# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))


def print_node(node_list):
    if len(node_list) == 0:
        return 

    while len(node_list) > 0:
        tmp = []
        nodes = []
        for node in node_list:
            tmp.append([node.val, node.left if node.left is None else node.left.val, node.right if node.right is None else node.right.val])
            if node.left is not None:
                nodes.append(node.left)
            if node.right is not None:
                nodes.append(node.right)
        print(tmp)
        node_list = nodes
    return 

def testCase():
    node_list = [1, 2, None, None, 3, 4, None, None, 5, None, None]
    codec = Codec()
    ret, _ = codec.getLeftRightNode(node_list, 0)
    print(ret.val)
    print_node([ret])

def testCase1():
    tn1 = TreeNode(1)
    tn2 = TreeNode(2)
    tn3 = TreeNode(3)
    tn4 = TreeNode(4)
    tn5 = TreeNode(5)
    tn1.left = tn2
    tn1.right = tn3
    tn3.left = tn4
    tn3.right = tn5
    codec = Codec()
    data = codec.serialize(tn1)

    node = codec.deserialize(data)
    print(data)
    print_node([node])


def testCase2():
    tn1 = TreeNode(1)
    tn2 = TreeNode(2)
    tn1.left = tn2
    codec = Codec()
    data = codec.serialize(tn1)
    print(data)
    node = codec.deserialize(data)
    print_node([node])

def testCase3():
    tn1 = TreeNode(1)
    tn2 = TreeNode(2)
    tn1.right = tn2
    codec = Codec()
    data = codec.serialize(tn1)
    # print(data)
    node = codec.deserialize(data)
    print_node([node])

[4,-7,-3,null,null,-9,-3, 9,-7,-4,null,6,null,-6,-6,null,null,0,6,5,null,9,null,null,-1,-4,null,null,null,-2]
def testCase4():
    node = TreeNode()
    tn1 = TreeNode(4)
    tn2 = TreeNode(-7)
    tn3 = TreeNode(-3)
    tn1.left = tn2
    tn1.right = tn3
    tn4 = TreeNode(-9)
    tn5 = TreeNode(-3)
    tn3.left = tn4 
    tn3.right = tn5



if __name__ == "__main__":
    testCase3()


