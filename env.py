from time import sleep
from nets import Net_list
from nodes import Node_list

import numpy as np

class Env():

    def __init__(self) -> None:

        '''
        为了简化布局复杂度，目前认为pin就是node本身
        因此不考虑pin相对于node的偏移坐标x_offset, y_offset

        同时，目前只选取5个net进行布局尝试
        '''

        self.nets = [] # the nets of pins
        self.nodes = [] # the list of all nodes' name, for adjacency matrix
        self.x = [] # the x coordinate of all nets
        self.y = [] # the y coordinate of all nets
        self.node_list = Node_list()
        net_list = Net_list()
        nets_name = net_list.netname_list
        for i in range(10): # choose 3 nets from net_list
            net = net_list.net_list[nets_name[i]]
            pin_list = net.pin_list
            nodes = []
            for pin in pin_list:
                nodes.append(pin.node)
                self.nodes.append(pin.node)
            x = []
            y = []
            for node in nodes:
                if node in self.node_list.movable_list.keys():
                    node_obj = self.node_list.movable_list[node]
                    x.append(node_obj.x)
                    y.append(node_obj.y)
                if node in self.node_list.fixed_list.keys():
                    node_obj = self.node_list.fixed_list[node]
                    x.append(node_obj.x)
                    y.append(node_obj.y)
                if node in self.node_list.overlap_list.keys():
                    node_obj = self.node_list.overlap_list[node]
                    x.append(node_obj.x)
                    y.append(node_obj.y)
            self.nets.append(nodes)
            self.x.append(x)
            self.y.append(y)
        # print(self.nets)    

    def reset(self):
        movable_list = self.node_list.movable_list
        fixed_list = self.node_list.fixed_list
        overlap_list = self.node_list.overlap_list

        s = []

        for node_name in self.nodes:
            if node_name in movable_list.keys():
                node = movable_list[node_name]
            if node_name in fixed_list.keys():
                node = fixed_list[node_name]
            if node_name in overlap_list.keys():
                node = overlap_list[node_name]
            xy = [node.x, node.y]
            s.append(xy)
            
        s = np.array(s)

        return s

    def step(self, s, a):

        '''
            parameters:
                s: state
                a: action from agent, an array of numpy like [x, y]
        '''

        # update state s
        for i in range(len(s)):
            if 0 in s[i]: # [x,y]
                s[i] = a
                break
        s_ = s
        
        # calculate reward
        r = 0
        for i in range(len(self.x)):
            x_max = max(self.x[i])
            x_min = min(self.x[i])
            y_max = max(self.y[i])
            y_min = min(self.y[i])
            HPWL = x_max - x_min + y_max - y_min
            r += -HPWL

        # set done
        if np.any(s_ == 0):
            done = False
        else:
            done = True
        return s_, r, done

    def render():
        pass

if __name__ == '__main__':
    import time
    start = time.time()

    env = Env()
    # state = env.reset()
    print(type(env.y))

    end = time.time()
    print(end - start) # running time, (s) 