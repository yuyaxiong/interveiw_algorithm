"""
定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头结点
"""
class ListNode(object):
    def __init__(self):
        self.value = None
        self.next = None

class Solution(object):
    def reverse_list(self, pHead):
        if pHead is None or pHead.next is None
            return pHead
        preNode = None
        pNode = pHead
        posNode = pNode.next
        while pNode.next is not None:
            pNode.next = preNode
            preNode = pNode
            pNode = posNode
            posNode = posNode.next
        pNode.next = preNode
        return pNode

class Solution1(object):
    def reverse_list(self, pHead):
        p_reversed_head = None
        p_node = pHead
        p_prev = None
        while p_node is not None:
            p_next = p_node.next
            if p_next is None:
                p_reversed_head = p_node
            p_node.next = p_prev
            p_prev = p_node
            p_node = p_next

        return p_reversed_head



