"""
请实现函数ComplexListNode* Clone(ComplexListNode* pHead), 复制一个复杂链表。在复杂链表中，每个节点除了有一个
m_pNext 指针指向下一个节点， 还有一个m_pSibling 指针指向链表中的任意节点或者nullptr。
    ......
   |     |
A->B->C->D->E
|  |  |     |
'''|''      |
   '''''''''

"""

class ComplexListNode(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.sibling = None

class Solution(object):
    def clone_node(self, pHead):
        pHead1 = pHead
        pHeadNew = ComplexListNode()
        pHeadNew1 = pHeadNew
        pHeadNew2 = pHeadNew
        oldToNew = {}
        while pHead1 is not None:
            pHeadNew.value = pHead.value
            pHead1 = pHead1.next
            pHeadNew.next = ComplexListNode()
            oldToNew[pHead1] = pHeadNew
        while pHead is not None:
            pHeadNew1.sibling = oldToNew[pHead.sibling]  
        return pHeadNew2




        
        
        
        
        
if __name__ == '__main__':
    s = Solution()
    


