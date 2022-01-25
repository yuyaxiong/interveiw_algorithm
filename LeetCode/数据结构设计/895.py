

#895. 最大频率栈
class FreqStack:

    def __init__(self):
        self.valToFreq = dict()
        self.freqToValDict = dict()
        self.max_freq = 0


    def push(self, val: int) -> None:
        freq = self.valToFreq.get(val, 0) + 1
        self.valToFreq[val] = freq
        if self.freqToValDict.get(freq) is None:
            self.freqToValDict[freq] = list()
        self.freqToValDict[freq].append(val)
        self.max_freq = max(self.max_freq, freq)
            

    def pop(self) -> int:
        val = self.freqToValDict.get(self.max_freq).pop()
        freq = self.valToFreq.get(val) -1
        self.valToFreq[val] = freq 
        if len(self.freqToValDict.get(self.max_freq)) == 0:
            del self.freqToValDict[self.max_freq]
            self.max_freq -= 1
        return val



# Your FreqStack object will be instantiated and called as such:
# obj = FreqStack()
# obj.push(val)
# param_2 = obj.pop()