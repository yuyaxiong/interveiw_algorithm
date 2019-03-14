# -*- coding:utf-8 -*-
"""
输入一个迷宫地图
-“#”表示墙，不可以进入，
-“.”表示可以进入， 且可以逗留
-"^"表示可以进入，一旦进入时空穿梭回原点(0,0)
-“￥”表示可以进入，一旦进入时空穿梭到终点（n-1, m-1）

请编码， 返回是否能够从（0,0）走到（n-1,m-1）。如果能够找到一条通路，返回TRUE，否则返回False.
如果（0,0）或者（n-1, m-1）位置为墙，返回False
"""

def create_maze():
  maze = [['.','.', '#','.', '#'], ['#', '.', '#', '^', '#'], ['#', '.', '#', '.', '#'], ['#', '$', '#', '.', '.']]
  return maze


# 思路就是将能遍历到的地方都以最短路径遍历到，如果终点没有遍历到，那么就不同，否则就通

def is_solvable(maze, n, m):
  maze_new = []
  for i in maze:
    maze_new.append([0 if j == '.' else j for j in i])
  if maze_new[n-1][m-1] == '#' or maze_new[0][0] == '#':
    return False
  cur_maze(maze_new, 0, 0, n, m, 0)
  return True if maze_new[n-1][m-1] > 0 else False

def cur_maze(maze, i, j,n,m, curstep):
  if i == n-1 and j == m-1:
    return
  curstep += 1

  # 上
  if i - 1 >= 0:
    if not isinstance(maze[i-1][j], str) and (maze[i-1][j] > curstep or maze[i-1][j] == 0):
      maze[i-1][j] = curstep
      cur_maze(maze, i-1, j, n, m, curstep)
    elif maze[i-1][j] == '^':
      cur_maze(maze, 0, 0, n, m, curstep)
    elif maze[i-1][j] == '$':
      maze[n - 1][m - 1] = curstep
      cur_maze(maze, n-1, m-1, n, m, curstep)

  # 下
  if i + 1 < n:
    if not isinstance(maze[i+1][j], str) and (maze[i+1][j] > curstep or maze[i+1][j] == 0):
      maze[i+1][j] = curstep
      cur_maze(maze, i+1, j, n, m, curstep)
    elif maze[i+1][j] == '^':
      cur_maze(maze, 0, 0, n, m, curstep)
    elif maze[i+1][j] == '$':
      maze[n - 1][m - 1] = curstep
      cur_maze(maze, n-1, m-1, n, m, curstep)

  # 右
  if j + 1 < m:
    if not isinstance(maze[i][j+1], str) and (maze[i][j+1] > curstep or maze[i][j+1] == 0):
      maze[i][j+1] = curstep
      cur_maze(maze, i, j+1, n, m, curstep)
    elif maze[i][j+1] == '^':
      cur_maze(maze, 0, 0, n, m, curstep)
    elif maze[i][j+1] == '$':
      maze[n - 1][m - 1] = curstep
      cur_maze(maze, n-1, m-1, n, m, curstep)

  # 左
  if j-1 >= 0:
    if not isinstance(maze[i][j-1], str) and (maze[i][j-1] > curstep or maze[i][j-1] == 0):
      maze[i][j-1] = curstep
      cur_maze(maze, i, j-1, n, m, curstep)
    elif maze[i][j-1] == '^':
      cur_maze(maze, 0, 0, n, m, curstep)
    elif maze[i][j-1] == '$':
      maze[n - 1][m - 1] = curstep
      cur_maze(maze, n-1, m-1, n, m, curstep)


if __name__ == "__main__":
  maze = create_maze()
  print(is_solvable(maze, 4, 5))