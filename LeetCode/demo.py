# Definition for singly-linked list.
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

node1 = ListNode(val=5)
node2 = ListNode(val=2)
node3 = ListNode(val=1)
node4 = ListNode(val=4)
# node1.next = node2
# node2.next = node3
# node3.next = node4

heap_list = [node1, node2, node3, node4]

heap_list = [(node1.val,node1), (node2.val, node2) , (node3.val, node3), (node4.val, node4)]
heapq.heapify(heap_list)

print(heapq.heappop(heap_list))


import queue

que = queue.Queue()
que.put(1)
que.put(2)
que.put(3)

while not que.empty():
    print(que.get())


class LFU:
    def __ini__(self):
        self.key = -1
        self.value = -1
        self.freq = 0
    
    def getFreq(self):
        return self.freq

    def getKey(self):
        return self.key

    def getValue(self):
        return self.value

    def setFreq(self, freq):
        self.freq = freq

    def setKey(self, key):
        self.key = key
        
    def setValue(self, value):
        self.value = value


class LFUCacheBak:
    def __init__(self, capacity: int):
        self.key_to_cache = dict()
        self.freq_to_order_key = dict()
        self.capacity = capacity
        self.min_freq = 0

    def get(self, key: int) -> int:
        if self.key_to_cache.get(key) is None:
            return -1
        cur_cache = self.key_to_cache.get(key)
        self.updateCache(cur_cache)
        return self.key_to_cache.get(key).value

    def put(self, key: int, value: int) -> None:

        if self.key_to_cache.get(key) is None:
            cache = LFU()
            cache.setKey(key)
            cache.setValue(value)
            self.balanceCache(cache)
        else:
            cur_cache = self.key_to_cache.get(key)
            cur_cache.setValue(value)
            self.updateCache(cur_cache)
            self.key_to_cache[key] = cur_cache

    def updateCache(self, lfu):
        if self.freq_to_order_key.get(lfu.getFreq()) is not None:
            if self.freq_to_order_key[lfu.getFreq()].get(lfu.getKey()) is not None:
                del self.freq_to_order_key[lfu.getFreq()][lfu.getKey()]
                if len(self.freq_to_order_key[lfu.getFreq()]) == 0:
                    del self.freq_to_order_key[lfu.getFreq()]
        
        lfu.setFreq(lfu.getFreq() + 1)
        if self.freq_to_order_key.get(lfu.getFreq()) is None:
            self.freq_to_order_key[lfu.getFreq()] = OrderedDict()
            self.freq_to_order_key[lfu.getFreq()][lfu.getKey()] = 1
        else:
            if lfu.getKey() in self.freq_to_order_key.get(lfu.getFreq()):
                del self.freq_to_order_key[lfu.getFreq()][lfu.getKey()]
            self.freq_to_order_key[lfu.getFreq()][lfu.getKey()] = 1 
        self.key_to_cache[lfu.getKey()] = lfu

    def balanceCache(self, lfu):
        if len(self.key_to_cache) == self.capacity:
            # ?
            min_freq = min(self.freq_to_order_key.keys())
            early_cache = self.freq_to_order_key.get(min_freq).pop() 
            del self.key_to_cache[early_cache.getKey()]

        self.key_to_cache[lfu.getKey()] = lfu
        self.updateCache(lfu)



