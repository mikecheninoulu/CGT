import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data_LPN import FAUST_DATA,SMG_DATA,SMPL_DATA
from objLoader_trimesh import trimesh_load_obj

import utils as utils
import numpy as np
import time
import trimesh
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Training NTP parameters')
parser.add_argument('--batch_size', type=int,default=8,help='training batch size')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle mesh points')
parser.add_argument('--model_type', type=str,default='original',help='model type')
parser.add_argument('--train_epoch', type=int,default=200,help='training epoch')
parser.add_argument('--train_size', type=int,default=400,help='training data size')
parser.add_argument('--dataset_name', type=str,default='SMG-3D',help='training data set')
parser.add_argument('--keep_train', type=int,default=0, help='keep training from checkpoint')
parser.add_argument('--lamda', type=float,default=0.0, help='center loss')


args = parser.parse_args()

batch_size = args.batch_size
shuffle_point = args.shuffle
train_epoch = args.train_epoch
train_size = args.train_size
dataset_name = args.dataset_name
keep_train = args.keep_train
lamda = args.lamda

model_type = args.model_type
if model_type == 'LPN':
    from model.model_LPN import NPT
elif model_type == 'LPN_deep':
    from model.model_LPN_deep import NPT
else:
    print('wrong model')

if dataset_name =='FAUST':
    dataset = FAUST_DATA(train=True, shuffle_point = shuffle_point, training_size = train_size)
elif dataset_name =='SMG-3D':
    dataset = SMG_DATA(train=True, shuffle_point = shuffle_point, training_size = train_size)
elif dataset_name =='NPT':
    dataset = SMPL_DATA(train=True, shuffle_point = shuffle_point, training_size = train_size)
elif dataset_name =='MG':
    dataset = MG_DATA(train=True, shuffle_point = shuffle_point, training_size = train_size)
elif dataset_name =='SMAL':
    dataset = SMAL_DATA(train=True, shuffle_point = shuffle_point, training_size = train_size)
else:
    print('wrong dataset')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)

model=NPT()
lrate=0.00005
optimizer_G = optim.Adam(model.parameters(), lr=lrate)
model.cuda()

print(keep_train)
if keep_train:
    checkpoint_path='./saved_model_LPN/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_lr.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Keeping training from epoch: ' + str(start_epoch))
else:
    model.apply(utils.weights_init)
    start_epoch = 0


scheduler = MultiStepLR(optimizer_G, milestones=[300,600], gamma=0.1)

print('training start')
print('Dataset:' + dataset_name)
print('Model:' + model_type)
print('Epoch:' + str(train_epoch))
print('Batch size:' + str(batch_size))
print('Sample size:' + str(train_size))
print('Shuffle point:' + str(shuffle_point))
print('Center loss Lamda:' + str(lamda))
loss_best = 0.2

for epoch in tqdm(range(start_epoch, train_epoch)):

    start=time.time()
    total_loss=0
    # switch model to evaluation mode
    model.train();
    '''training phase'''
    for j,data in enumerate(dataloader,0):

        optimizer_G.zero_grad()

        random_sample, gt_points, identity_points, new_face=data

        identity_points=identity_points.transpose(2,1)
        identity_points=identity_points.cuda()

        gt_points=gt_points.cuda()

        pointsReconstructed = model(identity_points)

        rec_loss = torch.mean((pointsReconstructed - gt_points)**2)
        # print('rec_loss')
        # print(rec_loss)

        edg_loss= 0
        for i in range(len(random_sample)):
            f=new_face[i].cpu().numpy()
            # print(f.shape)
            v=identity_points[i].transpose(0,1).cpu().numpy()
            # print(v.shape)
            edg_loss=edg_loss+utils.compute_score(pointsReconstructed[i].unsqueeze(0),f,utils.get_target(v,f,1))
        edg_loss=edg_loss/len(random_sample)
        # print('edg_loss')
        # print(edg_loss)

        central_distance_loss= 0
        for i in range(len(random_sample)):
            f=new_face[i].cpu().numpy()
            # print(f.shape)#(13776, 3)
            v=gt_points[i].unsqueeze(0)
            # print(v.shape)#(1,6890, 3)
            central_distance_loss += utils.central_distance_mean_score(pointsReconstructed[i].unsqueeze(0),v,f)
        central_distance_loss=central_distance_loss/len(random_sample)

        # print('central_distance_loss')
        # print(central_distance_loss)
        # print(a)

        l2_loss=rec_loss
        rec_loss=rec_loss+0.0005*edg_loss+lamda*central_distance_loss
        rec_loss.backward()
        optimizer_G.step()
        total_loss=total_loss+l2_loss

    print('####################################')
    # print(len(dataloader))
    print('Training')
    print('Epoch: ' +str(epoch))
    print(time.time()-start)
    mean_loss=total_loss/(j+1)
    print('Mean_loss',mean_loss.item())
    scheduler.step()
    print('####################################')


    # print(optimizer_G.param_groups[0]['lr'])
    if loss_best>mean_loss.item():
        loss_best = mean_loss.item()
        save_path='./saved_model_LPN/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_ep'+str(train_epoch)+'_lamda_'+str(lamda)+'.model'
        torch.save(model.state_dict(),save_path)

        checkpoint_path='./saved_model_LPN/'+dataset_name+'_type'+model_type+'_sf'+str(shuffle_point)+'_bs'+str(batch_size)+'_ts' + str(train_size) +'_ep'+str(train_epoch)+'_lamda_'+str(lamda)+'.pt'
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'loss': rec_loss,
                    }, checkpoint_path)
