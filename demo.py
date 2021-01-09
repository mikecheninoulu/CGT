import torch
from model.model_maxpool import NPT
import numpy as np
#import pymesh
# from objLoader import load_mesh
import trimesh
from objLoader_trimesh import trimesh_load_obj
net_G=NPT(num_points =28920 )
net_G.cuda()
net_G.load_state_dict(torch.load('./saved_model/maxpool.model'))


def face_reverse(faces):
    #id faces
    identity_faces=faces

    face_dict={}
    # 6890 samples
    # print(len(random_sample))
    # print('how')
    for i in range(len(random_sample)):
        #6890
        face_dict[random_sample[i]]=i
        # print(face_dict[random_sample[i]])
        # print('dont')
        # select the face_dic by random samples, make then equal to order
    new_f=[]
    # print('dict len')
    # print(face_dict)
    for i in range(len(identity_faces)):
        #try:
        new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            #print(face_dict[identity_faces[i][0]])
            # print(face_dict[identity_faces[i][1]])
            # print(face_dict[identity_faces[i][2]])
        #except:
            # print(face_dict[identity_faces[i][0]])
            # print(face_dict[identity_faces[i][1]])
            # print(identity_faces[i][0])
            # print(identity_faces[i][1])
            # print(identity_faces[i][2])
            #print('check')
            # print(face_dict[identity_faces[i][2]])
    # print(new_f)
    new_face=np.array(new_f)
    return new_face

#random
# random_sample = np.random.choice(6890,size=6890,replace=False)
# random_sample2 = np.random.choice(6890,size=6890,replace=False)
random_sample = np.random.choice(6890,size=6890,replace=False)
random_sample2 = np.random.choice(6890,size=6890,replace=False)
# print(random_sample)
# identity pose
id_mesh=trimesh_load_obj('./datasets/MG-cloth/Multi-Garment_dataset/125611510599246/smpl_registered.obj')
# expected pose
pose_mesh=trimesh_load_obj('./demo_data/14_664.obj')

with torch.no_grad():
    # get points of id
    # print(id_mesh.vertices)
    id_mesh_points=id_mesh.vertices[random_sample]
    # move to center
    #id_mesh_points-= id_mesh.center_mass
    # import to torch
    id_mesh_points = torch.from_numpy(id_mesh_points.astype(np.float32)).cuda()

    # get points of pose
    pose_mesh_points=pose_mesh.vertices#[random_sample2]
    # move to center
    #pose_mesh_points-= pose_mesh.center_mass
    # import to torch
    pose_mesh_points = torch.from_numpy(pose_mesh_points.astype(np.float32)).cuda()

    # reconstruct
    pointsReconstructed = net_G(pose_mesh_points.transpose(0,1).unsqueeze(0),id_mesh_points.transpose(0,1).unsqueeze(0))  # forward pass

# print('tell me why')
new_face=face_reverse(id_mesh.faces)
mesh = trimesh.Trimesh(vertices=pointsReconstructed.cpu().numpy().squeeze(),
                       faces=new_face)

mesh.export('./demo_data/demo.obj')
# pymesh.save_mesh_raw()
