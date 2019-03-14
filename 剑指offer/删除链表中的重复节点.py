"""
在一个排序的链表中，如何删除重复的节点？例如，在下图(a)中重复的几点被删除之后，链表如图(b)所示
(a). 1->2->3->4->5
(b). 1->2->5
"""

class ListNode(object):
    def __init__(self):
        self.value = None
        self.next = None

class Solution(object):
    def delete_duplicates(self, pHead):
        if pHead is None:
            return
        preNode = None
        pNode = pHead.next
        while pNode is not None:
            pNext = pNode.next
            needDelete = False
            if pNext is not None and pNext.value == pNode.value:
                needDelete = True
            if needDelete is False:
                preNode = pNode
                pNode = pNode.next
            else:
                value = pNode.value
                pToBeDelete = pNode
                while pToBeDelete is not None and pToBeDelete.value == value
                    pNext = pToBeDelete.next
                    pToBeDelete = pNext
                    # 如果是第一个点
                    if preNode is None:
                        pHead = pNext
                    else:
                        preNode.next = pNext
                    pNode = pNext

if __name__ == '__main__':
    pass