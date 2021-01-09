import trimesh
from trimesh.scene.scene import Scene
import logging
logger = logging.getLogger("pywavefront")
logger.setLevel(logging.ERROR)
import numpy as np

import meshio


class trimesh_load_obj_animal(object):
    def __init__(self, fileName):
        #self.bbox = np.zeros(shape=(2,3))
        ##
        self.vertices = []
        self.faces = []
        self.bbox = []

        objFile = open(fileName, 'r')

        for line in objFile:
            split = line.split()
            #print(split)
        	#if blank line, skip
            if split:
                if split[0] == 'f':
                    a = int(split[1].split('/')[0])-1
                    b = int(split[2].split('/')[0])-1
                    c = int(split[3].split('/')[0])-1
                    #     #f 492/1066-1/492 152/259/152 881/2223/881
                    self.faces.append([a,b,c])
            # else:
            #     print('haha')
            #     print(split)
            #     print(split.split(' ')[1])
            #
        # print(fileName)
        #obj_info = trimesh.load(fileName, file_type='obj', process=False,use_embree=False)
        meshio_mesh = meshio.read(fileName,file_format="obj")
        # print(obj_info)
        #self.vertices = obj_info.vertices
        self.vertices = meshio_mesh.points #- obj_info.center_mass
        self.vertices = np.array(self.vertices).astype("float32")
        self.vertices = self.vertices[:,[0,1,2]]

        self.faces = np.array(self.faces).astype("int32")
        #print(self.faces.shape)
        self.bbox = np.array([[np.max(self.vertices[:,0]), np.max(self.vertices[:,1]), np.max(self.vertices[:,2])], [np.min(self.vertices[:,0]), np.min(self.vertices[:,1]), np.min(self.vertices[:,2])]])
