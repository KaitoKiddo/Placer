import os
from file_cfg import File_cfg

class Pin:

    node = ''
    x_offset = 0
    y_offset = 0

    def __init__(self, node_name, x, y) -> None:
        self.node = node_name
        self.x_offset = x
        self.y_offset = y

class Net:

    name = '' # net name
    num = 0 # number of pins on this net
    pin_list = []

    def __init__(self, name, num) -> None:
        self.name = name
        self.num = num

    def add_pin(self, pin):
        self.pin_list.append(pin)


class Net_list:

    net_list = {}

    def __init__(self) -> None:
        file_cfg = File_cfg()
        dir = os.getcwd()
        nets_file = open(dir + '\\' + file_cfg.folder + '\\' + file_cfg.nets_filename)

        for i in range(7): # 过滤文件头
            temp = nets_file.readline()

        while True:
            line = nets_file.readline()
            line = line[:-1] # remove '\n'
            if len(line.split()) == 4:
                _, _, pin_num, net_name = line.split()
                pin_num = int(pin_num)
                net = Net(net_name, pin_num)
                for i in range(pin_num):
                    line = nets_file.readline()
                    line = line[:-1]
                    node_name, pin_direction, _, x_offset, y_offset = line.split()
                    x_offset = float(x_offset)
                    y_offset = float(y_offset)
                    pin = Pin(node_name, x_offset, y_offset)
                    net.add_pin(pin)
            else: # error
                pass

            self.net_list[net.name] = net # add net into net_list, {net_name : net class}

            if net_name == 'n990898':
                break

            # break

        nets_file.close()

if __name__ == '__main__':
    netlist = Net_list()
    print(netlist.net_list['n0'].pin_list[8].__dict__)

