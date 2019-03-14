
"""
输入数字n，按照顺序打印出1到最大的n位十进制数。比如输入3，则打印出1，2，3一直到最大的3位数999。

"""

class Solution(object):
    def print(self,n):
        if n <= 0:
            return
        number = [None for _ in range(n)]
        for i in range(10):
            number[0] = str(i)
            self.print_to_max_of_n_digits_recursively(number, n, 1)
        return

    def print_to_max_of_n_digits_recursively(self, number, length, index):
        if index > length - 1:
            # print(number)
            print(''.join(number))
            return
        for i in range(10):
            number[index] = str(i)
            self.print_to_max_of_n_digits_recursively(number, length, index+1)



if __name__=='__main__':
    s = Solution()
    s.print(3)