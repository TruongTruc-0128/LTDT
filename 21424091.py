from asyncio import PriorityQueue
from os import stat
from platform import node
import queue
from re import X
from typing import Tuple
import numpy as np
from queue import PriorityQueue as PQ # note queue is module
import networkx as nx

matrix_input_cu = np.array([[0, 0, 9, 3, 0, 0, 9, 0],
                            [0, 0, 8, 5, 5, 7, 6, 7],
                            [6, 9, 0, 4, 2, 5, 6, 8], 
                            [5, 1, 2, 0, 2, 4, 7, 9], 
                            [0, 3, 6, 8, 0, 6, 6, 5], 
                            [0, 6, 3, 9, 9, 0, 4, 0], 
                            [5, 1, 8, 1, 5, 3, 0, 5],  
                            [0, 3, 9, 8, 3, 0, 7, 0]])  # start: 1 - end: 6

matrix_SP6_cu = np.array([(0, 3, 7, 6, 9, 0), (3, 0, 8, 0, 3, 6), (5, 7, 0, 4, 0, 0),
              (9, 0, 8, 0, 6, 0),(7, 2, 0, 2, 0, 1),(0, 4, 0, 0, 8, 0)]) # start: 4 - end: 0

matrix_SP4 = np.array([[0, 5, 3, 0],
[6, 0, 0, 8],
[0, 5, 0, 0],
[6, 8, 4, 0]]) #start: 2 end: 1

matrix_SP5 = np.array([[0, 5, 2, 0, 7], 
[0, 0, 2, 0, 5], 
[0, 5, 0, 6, 2], 
[9, 3, 0, 0, 6], 
[0, 4, 9, 0, 0]]) #start: 1 end: 2

matrix_SP6 = np.array([[0, 7, 5, 0, 3, 0], 
[6, 0, 0, 7, 0, 8], 
[5, 9, 0, 4, 4, 0], 
[9, 0, 2, 0, 7, 0], 
[0, 5, 7, 0, 0, 2],
[0, 0, 6, 0, 0, 0]])  #start: 1 end: 0

matrix_SP7 = np.array([[0, 7, 9, 0, 3, 0, 9], 
[0, 0, 6, 0, 7, 3, 9], 
[7, 6, 0, 0, 6, 0, 4], 
[9, 0, 0, 0, 8, 5, 0], 
[4, 0, 0, 7, 0, 0, 0],
[8, 0, 9, 0, 0, 0, 0],
[3, 8, 0, 0, 0, 0, 0]]) #start: 1 end: 4


matrix_SP8 = np.array([[0, 6, 7, 0, 8, 0, 2, 0], 
                        [6, 0, 0, 4, 3, 4, 9, 1], 
                        [0, 6, 0, 0, 4, 4, 0, 0], 
                        [9, 9, 0, 0, 7, 0, 0, 1],
                        [0, 0, 5, 0, 0, 7, 0, 0],
                        [0, 8, 5, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 5, 5, 0, 0],
                        [7, 0, 0, 0, 6, 9, 8, 0]]) #start: 4 end: 3

matrix_input = np.array([
[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], 
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])

matrix_test = np.array([[0, 3, 7, 8],
                        [5, 0, 1, 1],
                        [1, 2, 0, 0],
                        [0, 0, 6, 0]]) #start: 1 end: 3

matrix_tu_tao = np.array([[0, 3, 7, 0, 9],
                            [0, 0, 1, 0, 2],
                            [2, 1, 0, 6, 0],
                            [2, 2, 0, 0, 8],
                            [0, 0, 0, 5, 0],])
                      
def BFS(matrix, start, end):
    """
    DFS algorithm
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:    
    vertices_total = len(matrix) # tổng các đỉnh trong đồ thị
    visited = {} # create list contains nodes is visited. note: key is vertex - value is vertex (default is -1)
    visited_nodes = []  # những nút đã thăm
    queue_frontier = []  # is queue (FIFO)
    path = [] #đường đi
    visited_nodes = [0 for _ in range(vertices_total)] # những nút nào được viến thì bật lên 1 (mặc định là 0)
    
    #begin:
    
    queue_frontier.append(start) 
    visited[start] = -1  # add value -1 is default
    visited_nodes[start] = 1
    # find neighborhoods to put into frontier list 
    while queue_frontier: # làm đến khi nào biên queue ko còn phần tử thì dừng lại
        i = 0
        if visited.get(end) is None:  # get trong dictionary để kiểm tra key truyền vào - # lấy từ visited ra lấy trúng end thì dừng
            while i < vertices_total: 
                # mở biên tại node hiện tại
                if matrix[start][i] != 0 and visited_nodes[i] != 1: # find neighbor nodes
                    queue_frontier.append(i) # add neighbor node into frontier
                    visited_nodes[i] = 1 # to mark nodes visited
                    i += 1
                else: i += 1

            # Sau khi mở biên xong sẽ thêm node cha vào visited và xóa nút đã mở biên(nút cha) đó và cập nhật lại nút tiếp theo trong queue
            if len(queue_frontier) > 1: 
                visited[queue_frontier[1]] = start  # gán dạng: {đỉnh cuối : đỉnh đầu}
                queue_frontier.pop(0)  # frontier.remove(matrix[start][i])  # remove fist element out of frontier  
                start = queue_frontier[0] # cập nhật start lại thành vị trí tiếp theo trong hàng đợi       
            else: 
                queue_frontier.pop(0)
        
            # print(frontier) # xuất ra/ cập nhật biên hiện tại 
        else: break

    for key in visited: # xuất đường đi
        path.append(key)
    print('BFS => ',end='')    
    return visited, path

# #test:
# BFS(matrix_SP4,2,1)

# BFS(matrix_SP5,1,2)
  
# BFS(matrix_SP6,1,0)
 
BFS(matrix_SP7,1,4)
 
# BFS(matrix_SP8,4,3)


def DFS(matrix, start, end):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO:        
    vertices_total = len(matrix)    
    path=[]
    visited={}
    stack_frontier = []
    visited_nodes = [0 for _ in range(vertices_total)]

    # note: duyệt đỉnh cuối có thự tự nhỏ nhất trước
    # assign for start node 
    visited[start] = -1
    stack_frontier.append(start) 
    visited_nodes[start] = 1

    while stack_frontier:
        i = 0 
        if visited.get(end) is None:
            while vertices_total > i:
                if matrix[start][i] != 0 and visited_nodes[i] != 1: # tìm neighbor node
                    stack_frontier.append(i)  # thêm neighbor node mới vào biên - tiếp theo sẽ dùng nút đó để mở nút kế   # điểm khác của DFS với BFS
                    visited_nodes[i] = 1
                    visited[i] = start   # gán dạng: {đỉnh cuối : đỉnh đầu}
                    start = i
                    break
                else: i += 1
                # dùng cấu trúc dữ liệu stack thay cho đệ quy
            if vertices_total <= i:
                if stack_frontier:
                    stack_frontier.pop() # default to remove last element out of frontier
                    start = visited[start] # gọi lại cha của start
                else: break
        else: break
    
    for p in visited: # xuất đường đi
        path.append(p)
    print('\nDFS => ',end='')      
    return visited, path

# DFS(matrix_SP4,2,1)

# DFS(matrix_SP5,1,2)
  
# DFS(matrix_SP6,1,0)
 
# DFS(matrix_SP7,1,6)
 
DFS(matrix_SP8,4,3)  
# DFS(matrix_SP8,4,7)

def UCS(matrix, start, end): # có dùng tuple
    """
    Uniform graph Search algorithm
     Parameters:visited
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:

    visited_nodes = [0]*len(matrix)
    path=[]
    visited={}                                         
    queue_frontier = PQ() # dùng Priority Queue cho queue_frontier (queue_frontier khác với queue_frontier BFS!)
    vertices_total = len(matrix) 
    
    # assgin default before runing
    print("\nGhi chú: một đỉnh/node gồm (a, b, c) tương ứng => (weight, cur_vertex, pas_vertex)\n")
    queue_frontier.put((0,start,start)) # datatype in put is Tuple to put incluces (weight, opened_vertex and old_vertex) in queue frontier. 
    visited[start] = -1 
    visited_nodes[start] = 1
    graph = queue_frontier.queue[0][0]
    print(queue_frontier.queue)

    while queue_frontier.queue:
        i = 0 
        if visited.get(end) is None:
            while vertices_total > i:
                if visited_nodes[i] != 1 and matrix[start][i] != 0: # find neighbourhood nodes (mở biên)
                    queue_frontier.put((matrix[start][i] + graph,i,start)) # to put incluces graph, opened_vertex, old_vertex in queue frontier (ví dụ: (12, 2, 5) đọc là chi phi đường đi thấp nhất tại đỉnh 2 đi từ đỉnh 5). 
                    # print(queue_frontier.queue)
                    i += 1
                else: i += 1
        
            # Sau khi mở biên xong sẽ thêm node cha vào visited và xóa nút đã mở biên(nút cha) đó và cập nhật lại nút tiếp theo trong queue
            if queue_frontier.qsize() > 1:
                cur_node = min(queue_frontier.queue)[1]
                pas_node = min(queue_frontier.queue)[2]
                visited[cur_node] = pas_node 
                print(visited)
                queue_frontier.get() # xóa node có chi phí đường đi nhỏ nhất
                print(queue_frontier.queue)
                while queue_frontier.queue:
                    start = min(queue_frontier.queue)[1] # gán start bằng node có chi phí đường đi nhỏ nhất mới
                    # kiểm tra đỉnh đó đã được thăm hay chưa                
                    if visited_nodes[start] == 1:
                        queue_frontier.get()
                        print(queue_frontier.queue)
                    else:
                        visited_nodes[start] = 1
                        min_weight = min(queue_frontier.queue)[0]
                        graph = min_weight # chi phí đường đi thấp nhất trước đó
                        break
            else: 
                queue_frontier.get()                
        else: break

    for e in visited:
        path.append(e)
    print("\nUCS => ",end='')
    return visited, path

UCS(matrix_SP8,4,3)

# print("Test")
# UCS(matrix_SP4,2,1)

# UCS(matrix_SP5,1,2)
  
# UCS(matrix_SP6,1,0)

# UCS(matrix_SP7,1,6)
 
# UCS(matrix_SP8,4,7)

# print("Test")

def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 
    # Decleration and set up default value
    path=[]
    visited={}
    visited_nodes = [0] * len(matrix)
    vertices_total = len(matrix)
    queue_frontier = PQ()

    visited[start] = - 1
    queue_frontier.put((0,start)) # put a Tuple includes 1st para is heuristic and 2nd is cur_vertex  
    print(queue_frontier.queue)
    print("\nGhi chú: một đinh/node gồm (a, b) tương ứng => (h, cur_vertex)\n")
    while queue_frontier.queue:
        i = 0
        if visited.get(end) is None: 
            while vertices_total > i:
                if matrix[start][i] != 0 and visited_nodes[i] != 1:
                    # to open frontier and put into queue_frontier
                    heuristic = matrix[start][i]
                    queue_frontier.put((heuristic,i))
                    i += 1
                    marked = True
                else: i += 1

            if marked:
                print(queue_frontier.queue)
            marked = False                
            print()
            # to remove vertex/node which has min heuristic from queue and then add into visited
            if queue_frontier.qsize() > 1:
                old_start = queue_frontier.get()
                visited[min(queue_frontier.queue)[1]] = old_start[1]
                while queue_frontier.queue:
                    start = min(queue_frontier.queue)[1]
                    if visited_nodes[start] == 1:
                        queue_frontier.get()
                    else: 
                        visited_nodes[start] = 1    
                        break
                print(queue_frontier.queue)
            else: queue_frontier.get()     
        else: break

    for i in visited:
        path.append(i)
    print("\nGBFS =>",end=' ')
    return visited, path

GBFS(matrix_tu_tao,3,2)

def initialize(matrix):
    '''Parameters
    -----------------------
         matrix: a numpy array stored adjacency matrix.
    -----------------------
    Return: 
        G: networkX graph.
        pos: vertice positions.
    '''
    n_vertices=matrix.shape[0]
    
    G=nx.DiGraph()
    for row in range(n_vertices):
        for col in range(n_vertices):
            w=matrix[row][col]
            if w!=0: G.add_edge(row, col, weight = matrix[row][col])

    pos = nx.spring_layout(G)  # positions for all nodes
    
    return pos
pos = initialize(matrix_SP8)

def printQueue(queue):
    q = queue 
    for node in q:
        print("(f:{f}, h:{h}, g:{g}, cv:{cv}, pv:{pv})".format(f = node[0], h = node[1], 
        g = node[2], cv = node[3], pv = node[4]), end=' ')
    print()
def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
     Parameters:
    ---------------------------
    matrix: np array UCS
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes 
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 
    '''
    note:
    DATA STRUCTURE IS IMPORTANT!
    f(n)=g(n)+h(n)
    f(n) = total estimated cost of path through node nn
    g(n) = cost so far to reach node nn
    h(n) = estimated cost from nn to goal. This is the heuristic part of the cost function, so it is like a guess.

    '''
    path=[]
    visited={}
    visited_nodes = [0] * len(matrix)
    vertices_total = len(matrix)
    queue_frontier = PQ()

    queue_frontier.put((0,0,0,start,start)) # includes graph/cost includes first f is estimate of total path cost,h is heuristic and g = cost of path so far, cur_vertex, past_vertex
    visited_nodes[start] = 1
    pos = initialize(matrix)
    g = 0 # graph
    # get x, y of end vertex
    x2 = pos[end][0]
    y2 = pos[end][1]    
    
    print("\nGhi chú: một đinh/node gồm (a, b, c, d, e) tương ứng => (f, h, g, cur_vertex, pas_vertex)\n")
    while queue_frontier.queue:
        if visited.get(end) is None:
            i = 0 
            while vertices_total > i:
                if matrix[start][i] != 0 and visited_nodes[i] != 1:
                    g += matrix[start][i] # get past path cost + current path cost  
                    # get x, y of current vertex
                    x1 = pos[i][0] 
                    y1 = pos[i][1]

                    # calculate heuristic:
                    h = round((abs(x2 - x1) + abs(y2 - y1)),2)
                    f = g + h
                    queue_frontier.put((round(f,2),h,g,i,start))
                    visited_nodes[i] = 1
                    marked = True  # state marked = 1 if queue is change (nếu queue thay đổi thì in ra)
                else: i += 1
            q = queue_frontier.queue
            if marked:
                print(q)
            marked = False
            # rid out of vertex which has min estimated cost total from queue and then add to vistied 
            if queue_frontier.qsize() > 1:
                #Xóa nút đã mở biên
                removed_node = queue_frontier.get()
                #Thêm vào visited
                visited[removed_node[3]] = removed_node[4]
                #kiểm tra đỉnh/node vừa xóa có còn liên quan trong queue không? thì xóa 
                for node in queue_frontier.queue:
                    if node[3] == removed_node[3]:
                        queue_frontier.get()
                start = min(queue_frontier.queue)[3]

                # printQueue(queue_frontier.queue)
                print(queue_frontier.queue)
                
            else: queue_frontier.get() 
                
        else: break

    for node in visited:
        path.append(node)

    print('\nA* =>', end=' ')
    return visited, path

Astar(matrix_SP8,4,3,pos)
# Astar(matrix_SP4,2,1,pos)
# Astar(matrix_SP5,1,2,pos)
# Astar(matrix_SP6,1,0,pos)
# Astar(matrix_SP7,1,4,pos)
