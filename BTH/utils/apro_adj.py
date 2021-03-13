import h5py
from calculate_neighbors import *
import scipy.io as sio
import args
# from . import args as args
from args import data_root,home_root

'''
We set tag=1 for closest pairs(similar),
tag=2 for pairs with middle distances(dissimilar),
tag = 0 for other cases (we don't care)
'''
with h5py.File(args.home_root+'data/latent_feats.h5','r') as h5_file: 
    video_feats = h5_file['feats'][:]  
with h5py.File(args.home_root+'data/anchors.h5','r') as h5_file:
    anchors = h5_file['feats'][:]  

Z,_,pos1 = ZZ(video_feats, anchors, 3, None)
s = np.asarray(Z.sum(0)).ravel()
isrl = np.diag(np.power(s, -1)) 
# isrl = inverse square root of lambda
Adj = np.dot(np.dot(Z,isrl),Z.T)
SS1 = (Adj>0.00001).astype('float32')

Z,_,pos1 = ZZ(video_feats, anchors, 4, None)
s = np.asarray(Z.sum(0)).ravel()
isrl = np.diag(np.power(s, -1)) 
Adj = np.dot(np.dot(Z,isrl),Z.T)
SS2 = (Adj>0.00001).astype('float32')

Z,_,pos1 = ZZ(video_feats, anchors, 5, None)
s = np.asarray(Z.sum(0)).ravel()
isrl = np.diag(np.power(s, -1)) 
Adj = np.dot(np.dot(Z,isrl),Z.T)
SS3 = (Adj>0.00001).astype('float32')

SS4 = SS3-SS2   
SS5 = 2*SS4+SS1

hh5 = h5py.File(args.data_root+'sim_matrix.h5', 'w')
hh5.create_dataset('adj', data = SS5)
hh5.close()