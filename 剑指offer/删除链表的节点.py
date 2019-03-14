"""
在O(1)时间内删除链表的节点。
给定单向链表的头指针和一个节点指针，定义一个函数在O(1)时间内删除该节点。

"""

class ListNode(object):
    def __init__(self):
        self.value = None
        self.next = None

class Solution(object):
    def delete_node(self, node, pHead):
        if node.next is not None:
            next_node = node.next
            node.value = next_node.value
            node.next = next_node.next
        elif node == pHead:
            pHead = None
            del node
        else:
            self.traver_list_node(pHead, node)
        return pHead

    def traver_list_node(self, pHead, node):
        preNode = None
        pNode = None
        while pNode != node:
            preNode = pHead
            pNode = pHead.next
        preNode.next = None
        del pNode
        return pHead

class Solution1(object):
    def delete_node(self,pHead, pToBeDeleted):
        if pHead is None or pToBeDeleted is None:
            return
        if pToBeDeleted.next is not None:
            pNext = pToBeDeleted.next
            pToBeDeleted.value = pNext.value
            pToBeDeleted.next = pNext.next

            pNext.next = None
            del pNext
        elif pHead == pToBeDeleted:
            del pToBeDeleted
            pHead = None
        else:
            pNode = pHead
            while pNode.next != pToBeDeleted:
                pNode = pNode.next
                
            pNode.next = None
            del pToBeDeleted