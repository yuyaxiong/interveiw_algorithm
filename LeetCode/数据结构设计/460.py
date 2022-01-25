
from collections import OrderedDict
# 460. LFU 缓存
class LFUCache:
    def __init__(self, capacity: int):
        self.key_to_value = dict()
        self.key_to_freq = dict()
        self.freq_to_order_key = dict()
        self.capacity = capacity
        self.min_freq = 0

        self.cum_time = 0 

    def get(self, key: int) -> int:
        if key not in self.key_to_value:
            return -1
        self.updateFreq(key)
        return self.key_to_value.get(key)

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return 0

        if self.key_to_value.get(key) is None:
            self.balanceFreq(key, value)
        else:
            self.key_to_value[key] = value
            self.updateFreq(key)


    def updateFreq(self, key):
        if self.key_to_freq.get(key) is None:
            self.key_to_freq[key] = 1
            if self.freq_to_order_key.get(1) is None:
                self.freq_to_order_key[1] = OrderedDict([(key, 1)])
            else:
                self.freq_to_order_key[1][key] = 1
            # 维护min_freq
            self.min_freq = 1
        else:
            # 更新freq, 删除老的freq
            cur_freq = self.key_to_freq[key]
            if self.freq_to_order_key.get(cur_freq) is not None:
                if key in self.freq_to_order_key.get(cur_freq):
                    del self.freq_to_order_key[cur_freq][key]
                    if len(self.freq_to_order_key.get(cur_freq)) == 0:
                        del self.freq_to_order_key[cur_freq]
                        # 维护min_freq
                        if cur_freq == self.min_freq:
                            self.min_freq = cur_freq + 1

            # 添加新的freq
            cur_freq += 1
            self.key_to_freq[key] = cur_freq
            if self.freq_to_order_key.get(cur_freq) is None:
                self.freq_to_order_key[cur_freq] = OrderedDict([(key, 1)])
            else:
                self.freq_to_order_key[cur_freq][key] = 1


    def balanceFreq(self, key, value):
        if len(self.key_to_value) == self.capacity:
            olden_key = self.getOrderDictTopKey(self.freq_to_order_key.get(self.min_freq))
            del self.freq_to_order_key.get(self.min_freq)[olden_key]
            if len(self.freq_to_order_key.get(self.min_freq)) == 0:
                del self.freq_to_order_key[self.min_freq]
            del self.key_to_value[olden_key]
            del self.key_to_freq[olden_key]

        self.key_to_value[key] = value 
        self.updateFreq(key)
            
    def getOrderDictTopKey(self, orderDict):
        for key in orderDict.keys():
            return key
                
 



# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


def testCase():
    cache = LFUCache(2)
    ret = [cache.put(1, 1), cache.put(2, 2), cache.get(1), 
    cache.put(3, 3), cache.get(2), cache.get(3), cache.put(4,4), 
    cache.get(1), cache.get(3), cache.get(4)]

    print(ret)

def testCase1():
    import time
    cache = LFUCache(10000)
    t1 = time.time()
    for i in range(100000):
        key, value = i, i * 5
        cache.put(key, value)
    for i in range(1000000):
        cache.get(key)
    t2 = time.time()
    delta = (t2 - t1) * 1000
    print(delta)
    print('cum_time:', cache.cum_time)
# [null,null,null,1,null,-1,3,null,-1,3,4]

    #285998.8663196564

    #25608.806371688843

if __name__ == "__main__":
    testCase1()