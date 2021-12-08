# CGN
Code for Contrastive-Geometry Transformer network for Generalized 3D Pose Transfer

## Dependencies

Requirements:
- python3.6
- numpy
- pytorch==1.1.0
- [trimesh]

Our code has been tested with Python 3.6, Pytorch1.1.0, CUDA 9.0 on Ubuntu 16.04.

## Data and Pre-trained model

### NPT dataset
We provide dataset and pre-trained model used, please download data from [data link](http://www.sdspeople.fudan.edu.cn/fuyanwei/download/NeuralPoseTransfer/data/), and download model weights from [model link](http://www.sdspeople.fudan.edu.cn/fuyanwei/download/NeuralPoseTransfer/ckpt/). The test data file lists are also provided, the mesh file order in file lists are `identiy pose gt`.
(Backup links: [Google Drive](https://drive.google.com/drive/folders/1ZduWjWn5sqbiU7aG2VSFm5YcdGudFTwk?usp=sharing))

### Other datasets

TBD

## Running the demo
We provide the pre-trained model for the original method and maxpooling method and also two meshes for test. For you own data, please train the model by yourself, because the pose parameter space may be different. For human meshes with clothes, we recommend the max-pooling method.
```
python demo.py
```

## Training
We provide both original and max-pooling methods. The original method has slightly better quantitative results. The max-pooling method is more convenient when dealing with identity and pose meshes with different number of vertices and this method produces smoother results.
```
python train.py
```
Tran LIR 
```
python train_latent_pose_normalization.py
```

## Evaluation
evaluation_NPT.py is the code for evaluation.

## Acknowledgement
TBD
