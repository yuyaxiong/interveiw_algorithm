from typing import List

# 视频拼接
class Solution:
    def videoStitching(self, clips: List[List[int]], time: int) -> int:
        clips = sorted(clips, key=lambda x: x[0])
        i = 0
        while i < len(clips) -1 and clips[i][0] == clips[i+1][0]:
            if clips[i][1] > clips[i+1][1]:
                clips[i], clips[i+1] = clips[i+1], clips[i]
            i += 1
        piv = clips[i]
        counter = 1
        if piv[0] == 0 and piv[1] >= time:
            return counter

        while i < len(clips):
            if piv[1] >= clips[i][0]:
                # print("cur_id:", i)
                if i < len(clips) -1:
                    while i < len(clips) -1 and clips[i+1][0] <= piv[1]:
                        if clips[i][1] > clips[i+1][1]:
                            clips[i], clips[i+1] = clips[i+1], clips[i] 
                        i += 1
                    piv = [piv[0], clips[i][1]]
                    counter += 1
                    if piv[0] == 0 and piv[1] >= time:
                        return counter
                    i += 1
                else:
                    piv = [piv[0], clips[i][1]]
                    counter += 1
                    i += 1
            else:
                return -1
        print(piv)
        if piv[0] == 0 and piv[1] >= time:
            return counter
        else:
            return -1

if __name__ == "__main__":
    # clips = [[0,1],[6,8],[0,2],[5,6],[0,4],[0,3],[6,7],[1,3],[4,7],[1,4],[2,5],[2,6],[3,4],[4,5],[5,7],[6,9]] 
    clips = [[5,7],[1,8],[0,0],[2,3],[4,5],[0,6],[5,10],[7,10]]
    print(len(clips))
    time = 5
    sol = Solution()
    ret = sol.videoStitching(clips, time)
    print(ret)