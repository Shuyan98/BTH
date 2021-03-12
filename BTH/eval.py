import numpy as np
from data import get_eval_loader
from args import *
import io
import time
import torch
import scipy.io as sio
from torch.autograd import Variable
import tools
import h5py
from args import *
from model import BERTLM

class Array():
    def __init__(self):
        pass
    def setmatrcs(self,matrics):
        self.matrics = matrics

    def concate_v(self,matrics):
        self.matrics = np.vstack((self.matrics,matrics))

    def getmatrics(self):
        return self.matrics


def evaluate(model, file_path, labels_name,num_sample):
    print 'loading test data...'
    hashcode = np.zeros((num_sample,nbits),dtype = np.float32)
    label_array = Array()
    hashcode_array = Array()
    rem = num_sample%test_batch_size
    labels = sio.loadmat(labels_name)['labels']
    eval_loader = get_eval_loader(file_path,batch_size=test_batch_size)
    label_array.setmatrcs(labels)
    
    batch_num = len(eval_loader)
    time0 = time.time()
    for i, data in enumerate(eval_loader): 
        data = {key: value.cuda() for key, value in data.items()}
        my_H,_,_ = model.forward(data["visual_word"])
        my_H = torch.mean(my_H,1)
        BinaryCode = torch.sign(my_H)
        if i == batch_num-1:
            hashcode[i*test_batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
        else:
            hashcode[i*test_batch_size:(i+1)*test_batch_size,:] = BinaryCode.data.cpu().numpy()

    test_hashcode = np.matrix(hashcode)
    time1 = time.time()
    print 'retrieval costs: ',time1-time0
    Hamming_distance = 0.5*(-np.dot(test_hashcode,test_hashcode.transpose())+nbits)
    time2 = time.time()
    print 'hamming distance computation costs: ',time2-time1
    HammingRank = np.argsort(Hamming_distance, axis=0)
    time3 = time.time()
    print 'hamming ranking costs: ',time3-time2

    labels = label_array.getmatrics()
    print 'labels shape: ',labels.shape
    sim_labels = np.dot(labels, labels.transpose())
    time6 = time.time()
    print 'similarity labels generation costs: ', time6 - time3

    records = open('./results/64_9288_2021.txt','w+')
    maps = []
    map_list = [5,10,20,40,60,80,100]
    for i in map_list:
        map,_,_ = tools.mAP(sim_labels, HammingRank,i)
        maps.append(map)
        records.write('topK: '+str(i)+'\tmap: '+str(map)+'\n')
        print 'i: ',i,' map: ', map,'\n'
    time7 = time.time()
    records.close()

def save_nf(model):
    '''
    To prepare latent video features, you can first train BTH model 
    with only mask_loss and save features with this function.
    '''
    num_sample = 45585 # number of training videos
    new_feats = np.zeros((num_sample,hidden_size),dtype = np.float32)
    rem = num_sample%test_batch_size
    eval_loader = get_eval_loader(train_feat_path,batch_size=test_batch_size)   
    batch_num = len(eval_loader)
    for i, data in enumerate(eval_loader): 
        data = {key: value.cuda() for key, value in data.items()}
        _,_,x = model.forward(data["visual_word"])
        feat = torch.mean(x,1)
        if i == batch_num-1:
            new_feats[i*test_batch_size:,:] = feat[:rem,:].data.cpu().numpy()
        else:
            new_feats[i*test_batch_size:(i+1)*test_batch_size,:] = feat.data.cpu().numpy()
    h5 = h5py.File(latent_feat_path, 'w')
    h5.create_dataset('feats', data = new_feats)
    h5.close()


if __name__ == '__main__':  
    model = BERTLM(feature_size).cuda()
    model.load_state_dict(torch.load(file_path + '/9288.pth'))
    h5_file = h5py.File(test_feat_path, 'r')
    video_feats = h5_file['feats']
    num_sample = len(video_feats)
    print num_sample
    model.eval()
    evaluate(model, test_feat_path, label_path ,num_sample)
    # save_nf(model)
