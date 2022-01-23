

# 
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if self.cache.get(key) is None:
            return -1
        self.makeRecentKey(key)
        return self.cache.get(key)

    def put(self, key: int, value: int) -> None:
        if self.cache.get(key) is not None:
            self.cache[key] = value
            self.makeRecentKey(key)
            return 
        if len(self.cache) >= self.capacity:
            early_key = list(self.cache.keys())[0]
            del self.cache[early_key]
        self.cache[key] = value

    def makeRecentKey(self, key: int) -> None:
        val = self.cache.get(key)
        del self.cache[key]
        self.cache[key] = val



# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

def testCase():
    capacity = 2
    lru = LRUCache(capacity)
    # ret = [lru.put(1, 1) ,lru.put(2, 2), lru.get(1), lru.put(3, 3), lru.get(2), lru.put(4, 4), lru.get(1), lru.get(3), lru.get(4)]
    ret1 = [lru.put(2, 1), lru.put(2, 2), lru.get(2) ,lru.put(1, 1), lru.put(4, 1), lru.get(2)]
    print(ret1)


if __name__ == "__main__":
    testCase()