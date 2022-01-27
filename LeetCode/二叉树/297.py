# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return ''
        ser_list = []
        self.serHelp(root, ser_list)
        ser_list = [str(s) for s in ser_list]
        return "|".join(ser_list)

    def serHelp(self, root, ser_list):
        if root is None:
            ser_list.append('None')
            return 
        ser_list.append(root.val)
        self.serHelp(root.left, ser_list)
        self.serHelp(root.right, ser_list)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if len(data) == 0:
            return None
        ser_list = [None if d == 'None' else int(d) for d in data.split("|")]
        node, i = self.getLeftRightNode(ser_list, 0)
        return node

    def getLeftRightNode(self, ser_list, i):
        if i == len(ser_list)-1:
            node = ser_list[i]
            return node, i+1

        node = TreeNode(ser_list[i])
        if ser_list[i+1] is None:
            if ser_list[i+2] is None:
                i += 3
                return node, i
            else:
                i += 2
                node.right, i = self.getLeftRightNode(ser_list, i)
                return node, i
        else:
            i += 1
            node.left, i = self.getLeftRightNode(ser_list, i)
            node.right, i = self.getLeftRightNode(ser_list, i)
            return node, i                


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


if __name__ == "__main__":
    testCase3()


