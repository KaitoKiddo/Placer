from networkx.algorithms.distance_measures import eccentricity
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

        net_list = Net_list()
        nets_name = net_list.netname_list
        for i in range(5): # choose 3 nets from net_list
            net = net_list.net_list[nets_name[i]]
            pin_list = net.pin_list
            nodes = []
            for pin in pin_list:
                nodes.append(pin.node)
                self.nodes.append(pin.node)
            self.nets.append(nodes)
        # print(self.nets)    

        self.node_list = Node_list()

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

        s_ = s

        r = 1

        return s_, r

if __name__ == '__main__':
    import time
    start = time.time()

    env = Env()
    # state = env.reset()
    print(env.nets)

    end = time.time()
    print(end - start) # running time, (s)