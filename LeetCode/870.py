class Solution:
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if len(nums1) != len(nums2):
            return []
        # 排序
        nums1_s = sorted(nums1)
        nums2_s = sorted(nums2)
        n2_dict = dict()
        # nums2的词典记录索引位置
        for i, n in enumerate(nums2):
            if n2_dict.get(n) is None:
                n2_dict[n] = []
            n2_dict[n].append(i)
        i, j = 0, 0
        nums1_tmp = [None for _ in nums1]
        # 排序后双指针对比
        while i < len(nums1) and j < len(nums2):
            if nums1_s[i] <= nums2_s[j]:
                i += 1
            else:
                nums1_tmp[j] = nums1_s[i]
                nums1_s[i] = None
                i += 1
                j += 1

        nums1_s = list(filter(lambda x: x != None, nums1_s))
        # 补数
        while j < len(nums2):
            nums1_tmp[j] = nums1_s.pop()
            j += 1
        # 还原
        for i, n in enumerate(nums1_tmp):
            n2_idx = n2_dict.get(nums2_s[i]).pop()
            nums1[n2_idx] = n
        return nums1




