import torch
import numpy as np
import trimesh
from objLoader_trimesh import trimesh_load_obj

model_path = './saved_model/good/CGP_split4_sfTrue_bs8_ts8000_ep1000_lr_0.00006.model'

model_type = model_path.split('/')[-1].split('_sf')[0].split('type')[1]

print(model_type)
if model_type == 'original':
    from model.model import NPT
elif model_type == 'max_pool':
    from model.model_maxpool import NPT
elif model_type == 'CGP':
    from model.model_CGP import NPT
elif model_type == 'max_pool_CGP':
    from model.model_maxpool_CGP import NPT
else:
    print('wrong model')
best_score = 1.0
net_G=NPT()
net_G.cuda()
net_G.load_state_dict(torch.load(model_path))

random_sample = np.random.choice(6890,size=6890,replace=False)
random_sample2 = np.random.choice(6890,size=6890,replace=False)
#data_path = './datasets/NPT/npt_data/'
#setting = ['supervised_list_obj', 'unsupervised_list_obj']
#for set in setting:
    #print(set +' evaluation starts:')
    #list_path = './datasets/NPT/list/'+set+'.txt'
    #list = open(list_path,'r').read().splitlines()

data_path = './datasets/NPT/npt_data/'
setting = ['supervised_list_obj', 'unsupervised_list_obj']
for set in setting:
    print(set +' evaluation starts:')
    list_path = './datasets/NPT/list/'+set+'.txt'
    list = open(list_path,'r').read().splitlines()
    rec_loss = 0.0
    for pairs in list:
        id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
        id_mesh = trimesh_load_obj(data_path + id_mesh_name)
        pose_mesh=trimesh_load_obj(data_path+ pose_mesh_name)
        gt_mesh=trimesh_load_obj(data_path+ gt_mesh_name)
        with torch.no_grad():

            id_mesh_points=id_mesh.vertices[random_sample2]
            id_mesh_points = id_mesh_points - (id_mesh.bbox[0] + id_mesh.bbox[1]) / 2
            id_mesh_points = torch.from_numpy(id_mesh_points.astype(np.float32)).cuda()
            #print(id_mesh_points.shape)

            pose_mesh_points=pose_mesh.vertices[random_sample]
            pose_mesh_points = pose_mesh_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
            pose_mesh_points = torch.from_numpy(pose_mesh_points.astype(np.float32)).cuda()

            # ground truth
            gt_mesh_points=gt_mesh.vertices[random_sample2]
            gt_mesh_points = gt_mesh_points - (gt_mesh.bbox[0] + gt_mesh.bbox[1]) / 2

            # reconstruct
            pointsReconstructed = net_G(pose_mesh_points.transpose(0,1).unsqueeze(0),
            id_mesh_points.transpose(0,1).unsqueeze(0))  # forward pass
            pointsReconstructed = pointsReconstructed.cpu().numpy().squeeze()
            bbox = np.array([[np.max(pointsReconstructed[:,0]), np.max(pointsReconstructed[:,1]),
            np.max(pointsReconstructed[:,2])], [np.min(pointsReconstructed[:,0]),
            np.min(pointsReconstructed[:,1]), np.min(pointsReconstructed[:,2])]])
            pointsReconstructed = pointsReconstructed - (bbox[0] + bbox[1]) / 2

            rec_loss =  rec_loss + np.mean((pointsReconstructed - gt_mesh_points)**2)

    score_final = rec_loss / len(list)
    print(model_path.split('/')[-1])
    print('The score for ' + set + ' is: ' +str(score_final))
