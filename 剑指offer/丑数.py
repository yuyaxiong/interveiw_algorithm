"""
我们把值包含因子2，3，5的数称为丑数。求按从小到打的顺序的第1500个丑数。
例如：6，8都是丑数，但14不是，因为它包含因子7。
习惯上我们把1当做第一个丑数。
"""
class Solution(object):
    def is_ugle(self, num):
        while num % 2 == 0:
            num = num // 2
        while num % 3 == 0:
            num = num // 3
        while num % 5 == 0:
            num = num // 5
        return True if num == 1 else False

    def find_ugle(self, n):
        i = 1
        counter = 1
        while counter < n:
            if self.is_ugle(i):
                counter +=1
            i += 1
        return i

class Solution1(object):
    def find_ugle(self, num):
        tmp = [1, 2, 3, 4, 5]
        if num <= 0:
            return 0
        elif num <= 5 and num>0 :
            return tmp[num-1]

        T2, T3, T5 = 2, 1, 1
        counter = 5
        while counter < num:
            ulge_value = min(tmp[T2]*2, tmp[T3] *3, tmp[T5]*5)
            # 这一段很重要
            while ulge_value >= tmp[T2]*2:
                T2 += 1
            while ulge_value >= tmp[T3]*3:
                T3 += 1
            while ulge_value >= tmp[T5]*5:
                T5 += 1
            tmp.append(ulge_value)
            counter += 1
        return tmp[-1]



if __name__ == "__main__":
    s = Solution1()
    print(s.find_ugle(0))
    # print(s.is_ugle(1))