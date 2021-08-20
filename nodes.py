import os
from file_cfg import File_cfg

class Node:
    
    n = '' # name
    w = 0 # width
    h = 0 # height
    t = 0 # movetype 0:movable 1:fixed 2:fixed but can overlap

    x = 0 # X
    y = 0 # Y

    def __init__(self, n, w, h, t) -> None:
        self.n = n
        self.w = w
        self.h = h
        self.t = t

    # 节点的操作，修改坐标
    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y


class Node_list:

    movable_list = {}
    fixed_list = {}
    overlap_list = {}

    def __init__(self) -> None:
        file_cfg = File_cfg()
        dir = os.getcwd()

        # process nodes file
        nodes_file = open(dir + '\\' + file_cfg.folder + '\\' + file_cfg.nodes_filename)
        
        for i in range(7): # 把文件开头过滤掉
            temp = nodes_file.readline() 
        
        for line in nodes_file.readlines(): # 开始读取文件节点信息
            line = line[:-1] # remove '\n'
            if len(line.split()) == 3: # movable node
                n, w, h = line.split()
                w = int(w)
                h = int(h)
                t = 0
                node = Node(n, w, h, t)
                self.movable_list[n] = node
            elif len(line.split()) == 4: # fixed node
                n, w, h, t = line.split()
                w = int(w)
                h = int(h)
                if t == 'terminal': # can't overlap
                    t = 1
                    node = Node(n, w, h, t)
                    self.fixed_list[n] = node
                elif t == 'terminal_NI': # overlap
                    t = 2
                    node = Node(n, w, h, t)
                    self.overlap_list[n] = node
                else: # error
                    pass
            else: # error
                pass
            # break
        nodes_file.close()

        # process pl file
        pl_file = open(dir + '\\' + file_cfg.folder + '\\' + file_cfg.pl_filename)

        for i in range(4): # 过滤文件头
            temp = pl_file.readline()

        for line in pl_file.readlines():
            line = line[:-1]
            if len(line.split()) == 5:
                name, x, y, _, _ = line.split()
                x = int(x)
                y = int(y)
                node = self.movable_list[name]
                node.set_x(x)
                node.set_y(y)
            elif len(line.split()) == 6:
                name, x, y, _, _, type = line.split()
                x = int(x)
                y = int(y)
                if type == '/FIXED':
                    node = self.fixed_list[name]
                    node.set_x(x)
                    node.set_y(y)
                elif type == '/FIXED_NI':
                    node = self.overlap_list[name]
                    node.set_x(x)
                    node.set_y(y)
                else: # error
                    pass
            else: # error
                pass
            
            print(name)
            # break

        pl_file.close()


if __name__ == '__main__':
    # node1 = Node('o0', 5, 9, 1)
    nodelist = Node_list()
    print(nodelist.overlap_list['p16391'].__dict__)