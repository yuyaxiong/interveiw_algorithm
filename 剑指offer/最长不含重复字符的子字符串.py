"""
请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长字符串的长度。
假设字符串中只包含‘a’-'z'的字符。例如，在字符串"arabcacfr"中，最长的不含
重复字符的子字符串是"acfr"，长度为4。
"""

class Solution(object):
    def longest_substring_without_duplication(self, strings):
        s, e = 0, 0
        length = 0
        for idx in range(len(strings)):
            substrs = strings[s: idx]
            current = strings[idx]
            if current not in substrs:
                length += 1
            else:
                last = self.first_idx(strings[s:idx], current)
                if length > idx - last:
                    length = idx - last
                    s = last + 1
                else:
                    length += 1
        return length

    def first_idx(self, strings, strs):
        for i in range(len(strings)):
            if strings[i] == strs:
                return i
                

if __name__ == '__main__':
    s = Solution()
    strings = 'arabcacfr'
    print(s.longest_substring_without_duplication(strings))