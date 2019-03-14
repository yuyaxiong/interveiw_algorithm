"""
给定一个只含英文字母a-z的字符串，返回该字符串中长度最长的回文子串，如果存在多个返回任意一个。
回文串是指从左到右，和从右往左读一致的字符串，比如：aba,abba。
例如：
输入 input_text: "abaccabac"
输出 output_text: "abaccaba"
"""

def longestPalindrome(s):
    # 基本思路是对任意字符串，如果头和尾相同，那么它的最长回文子串一定是去头去尾之后的部分的最长回文子串加上头和尾。
    # 如果头和尾不同，那么它的最长回文子串是去头的部分的最长回文子串和去尾的部分的最长回文子串的较长的那一个。
    n = len(s)
    maxl = 0
    start = 0
    # 这个方法相当于相对于暴力搜索而言剪枝了
    for end in range(n):
        # 第一个if 是两头都+1后是否是回文子串， 第二个if 是只要一头+1后是否是回文子串
        if end - maxl >= 1 and isPalindrome(s[end-maxl-1:end+1]):
            start = end - maxl - 1
            maxl += 2 # 因为start减去了1， end 加了1，所以长度增加2
            continue
        if end - maxl >= 0 and isPalindrome(s[end-maxl: end+1]):
            start = end - maxl
            maxl += 1 # 因为start没有减去1，end加了1，所以长度加1
    return s[start: start + maxl]

def isPalindrome(s):
    start, end = 0, len(s)-1
    while start < end:
        if s[start] == s[end]:
            start += 1
            end -= 1
        else:
            return False
    return True

if __name__ == '__main__':
    s = 'abcbaddd'
    print(longestPalindrome(s))
