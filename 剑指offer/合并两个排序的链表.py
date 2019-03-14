"""
输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
例如， 输入图3.11中的链表1和链表2，则合并之后的升序链表如链表3所示。
1->3->5->7
2->4->6->8
1->2->3->4->5->6->7->8
"""

class ListNode(object):
    def __init__(self):
        self.value = None
        self.next = None

class Solution(object):
    def merge_list(self, pHead1, pHead2):
        if pHead1 is None:
            return pHead2
        if pHead2 is None:
            return pHead1
        pHead = ListNode()
        pNode = pHead
        while pHead1 is not None and pHead2 is not None:
            if pHead1.value > pHead2.value:
                pNode.next = pHead2
                pNode = pNode.next
                pHead2 = pHead.next
            else:
                pNode.next = pHead1
                pNode = pNode.next
                pHead1 = pHead1.next

        if pHead1 is not None:
            pNode.next = pHead1
        if pHead2 is not None:
            pNode.next = pHead2
        return pHead.next

class Solution1(object):
    def merge(self, pHead1, pHead2):
        if pHead1 is None:
            return pHead2
        if pHead2 is None:
            return pHead1
        pMergeHead = None
        if pHead1.value < pHead2.value:
            pMergeHead = pHead1
            pMergeHead.next= self.merge(pHead1.next, pHead2)
        else:
            pMergeHead = pHead2
            pMergeHead.next = self.merge(pHead1, pHead2.next)

        return pMergeHead


