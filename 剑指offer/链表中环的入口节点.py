"""
如果一个链表中包含环，如何找出环的入口节点？例如，在如下图所示的链表中，环的入口节点为3

1->2->3->4->5->6
      |________|

"""

class ListNode(object):
    def __init__(self):
        self.value = None
        self.next = None

class Solution(object):
    def meeting_node(self, pListHead):
        if pListHead is None:
            return None

        exists_flag, clcye_num = self.is_existis_cycyle(pListHead)
        if exists_flag:
            counter = 0
            fp = pListHead
            while counter < clcye_num:
                fp = fp.next
                counter += 1

            ep = pListHead
            while fp != ep:
                fp = fp.next
                ep = ep.next
            return fp
        else:
            return None


    def is_existis_cycyle(self, pListHead):
        fp = pListHead
        sp = pListHead
        while sp != fp:
            if fp is None:
                return False,0
            fp = fp.next
            if fp is not None:
                fp = fp.next
            sp = sp.next

        counter = 0
        while fp != sp:
            counter += 1
            fp = fp.next
        return True, counter-1