import os
		
class File_cfg:

	folder = ''
	nodes_filename = ''
	nets_filename = ''
	pl_filename = ''
	scl_filename = ''
	shapes_filename = ''
	route_filename = ''
	wts_filename = ''

	def __init__(self) -> None:

		dir = os.getcwd()
		files = os.listdir(dir)
		for i in files:
			if os.path.isdir(i):
				if '.' in i:
					continue
				if '_' in i:
					continue
				else:
					self.folder = i
					break
		# print(dir)

		aux_file = open(dir + '/' + self.folder + '/' + self.folder + '.aux')
		aux = aux_file.readline()
		aux_file.close()
		aux = aux.split()
		for i in aux:
			if '.nodes' in i:
				self.nodes_filename = i
			if '.nets' in i:
				self.nets_filename = i
			if '.pl' in i:
				self.pl_filename = i
			if '.route' in i:
				self.route_filename = i 
			if '.scl' in i:
				self.scl_filename = i 
			if '.shapes' in i:
				self.shapes_filename = i 
			if '.wts' in i:
				self.wts_filename = i


if __name__ == '__main__':
	file = File_cfg()
	
	print(file.nodes_filename, file.nets_filename, file.pl_filename, file.route_filename, file.scl_filename, file.shapes_filename, file.wts_filename)
