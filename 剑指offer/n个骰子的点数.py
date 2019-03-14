"""
n个骰子的点数
把n个骰子仍在地上，所有骰子朝上一面的点数之和为s。
输入n，打印出s的所有可能的值出现的概率。
"""
class Solution(object):
    MAX_VALUE = 6
    def print_probability(self, num):
        if num < 1:
            return
        max_sum = num * self.MAX_VALUE
        prob = [0 for _ in range(max_sum - num + 1)]
        self.probability(num, prob)
        total = self.MAX_VALUE ** num
        for idx, p in enumerate(prob):
            print(idx + num, p, p/total)

    def probability(self, num, prob):
        for i in range(1, self.MAX_VALUE):
            self.probability_help(num, num, i, prob)

    def probability_help(self, original, current, sum, prob):
        if current == 1:
            prob[sum - original] += 1
        else:
            for i in range(1, self.MAX_VALUE):
                self.probability_help(original, current-1, i+sum, prob)

class Solution1(object):
    MAX_VALUE = 6
    def print_probability(self, num):
        if num < 1:
            return
        prob = [[0 for _ in range(self.MAX_VALUE * num +1)] for _ in range(2)]
        flag = 0
        # 初始化
        for i in range(1, self.MAX_VALUE+1):
            prob[flag][i] = 1
        # 迭代
        for k in range(2, num+1):
            for i in range(0, k):
                prob[1-flag][i] = 0

            for i in range(k, self.MAX_VALUE * k+1):
                prob[1-flag][i] = 0
                j = 1
                # f2(n) = f1(n-1) + f1(n-2) + f1(n-3) + f1(n-4) + f1(n-5) + f1(n-6)
                while j <= i and j <= self.MAX_VALUE:
                    j += 1
                    prob[1-flag][i] += prob[flag][i-j]
            flag = 1 - flag

        total = self.MAX_VALUE ** num
        for i in range(num, self.MAX_VALUE * num+1):
            print(i, prob[flag][i], prob[flag][i]/total)



if __name__ == '__main__':
    s = Solution1()
    s.print_probability(6)