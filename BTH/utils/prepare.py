import h5py
from calculate_neighbors import *
from args import train_assist_path, anchor_path,latent_feat_path

'''
For each video, search several neighbors from the anchor set and save them in a file.
To save space, we only save the index of them.
The nearest anchor is a pseudo label of the video.
'''
with h5py.File(latent_feat_path,'r') as h5_file:
    train_feats = h5_file['feats'][:]  
with h5py.File(anchor_path,'r') as h5_file:
    anchors = h5_file['feats'][:]   

Z1,_,pos1 = ZZ(train_feats, anchors, 3, None)
h5 = h5py.File(train_assist_path, 'w')
h5.create_dataset('pos', data = pos1)
h5.close()