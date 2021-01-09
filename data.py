import torch.utils.data as data
import torch
import numpy as np
import trimesh
from objLoader_trimesh import trimesh_load_obj
from objLoader_trimesh_animal import trimesh_load_obj_animal
import random

class SMPL_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/NPT/npt_data/'
        self.length = training_size
        self.test_label_path = './datasets/NPT/list/supervised_list_obj.txt'
        self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            identity_mesh_i=np.random.randint(0,16)
            identity_mesh_p=np.random.randint(200,600)

            pose_mesh_i=np.random.randint(0,16)
            pose_mesh_p=np.random.randint(200,600)

            identity_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj'
            # print(identity_mesh_path)
            pose_mesh_path =self.path+'id'+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'.obj'
            gt_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'.obj'
        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)


class SMG_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/SMG/SMG_3d_rotated/'
        self.length = training_size
        self.test_label_path = './datasets/SMG/MG_list/seen_pose_list.txt'
        self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            identity_mesh_i=np.random.randint(0,35)
            identity_mesh_p=np.random.randint(0,180)

            pose_mesh_i=np.random.randint(0,35)
            pose_mesh_p=np.random.randint(0,180)

            identity_mesh_path = self.path+'id_'+str(identity_mesh_i)+'_pose_'+str(identity_mesh_p)+'.obj'
            # print(identity_mesh_path)
            pose_mesh_path =self.path+'id_'+str(pose_mesh_i)+'_pose_'+str(pose_mesh_p)+'.obj'
            gt_mesh_path = self.path+'id_'+str(identity_mesh_i)+'_pose_'+str(pose_mesh_p)+'.obj'

        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)



class SMAL_DATA(data.Dataset):
    def __init__(self, train,  npoints=3889, shuffle_point = False, training_size = 16):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/SMAL/processed/'
        self.length = training_size
        self.train_label_path = './datasets/SMAL/processed/train_list_lion_cow.txt'
        self.train_list = open(self.train_label_path,'r').read().splitlines()

        self.test_label_path = './datasets/SMG/MG_list/seen_pose_list.txt'
        self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            pairs = self.train_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')

            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name
        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name


        identity_mesh=trimesh_load_obj_animal(identity_mesh_path)
        pose_mesh=trimesh_load_obj_animal(pose_mesh_path)
        gt_mesh=trimesh_load_obj_animal(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)


#FAUST
class FAUST_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/FAUST/MPI-FAUST/processed_FAUST/FAUST_body/'
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            identity_mesh_i=np.random.randint(0,5)
            identity_mesh_p=np.random.randint(0,10)

            gt_mesh_i=identity_mesh_i
            gt_mesh_p=0

            identity_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj'
            # print(identity_mesh_path)

            gt_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(gt_mesh_p)+'.obj'
        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name

        #print(identity_mesh_path)
        #print(gt_mesh_path)
        identity_mesh=trimesh_load_obj(identity_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)
