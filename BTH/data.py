import pickle
import h5py
import torch
import torch.utils.data as data
import numpy as np
import random
from utils.args import train_assist_path,anchor_path


class TrainDataset(data.Dataset):

    def __init__(self, feature_h5_path, sim_path):
        
        with h5py.File(feature_h5_path,'r') as h5_file:
          self.video_feats = h5_file['feats'][:]   
        with h5py.File(sim_path,'r') as h5_file:
          self.v_feats = h5_file['adj'][:]   
        with h5py.File(train_assist_path,'r') as h5_file:
          self.neighbor = h5_file['pos'][:] 
        with h5py.File(anchor_path,'r') as h5_file:
          self.achors = h5_file['feats'][:]  
        

    def random_frame(self, video):
        output_label = np.zeros((video.shape),dtype=np.float32)
        for i, token in enumerate(video):
            prob = random.uniform(0,1)
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    video[i] = video[i]*0

                # 10% randomly change token to random token
                elif prob < 0.9:
                    xx = np.random.randint(0,self.video_feats.shape[0],size=1)
                    yy = np.random.randint(0,self.video_feats.shape[1],size=1)
                    video[i] = self.video_feats[xx[0],yy[0],:]
            output_label[i] = token

        return video, output_label

    def random_video(self, index):
        if random.random() > 0.5:
            # get a similar video
            index2 = self.get_neighbor_video(index)
            return index2, 1.0
        else:
            # get a dissimilar video
            index2 = self.get_random_video(index)
            return index2, -1.0

    def get_neighbor_video(self, index):
        # similar videos have closest distance from the query video (tag is 1)
        neighbors = np.where(self.v_feats[index]==1)
        neighbors=neighbors[0]
        ranind = np.random.randint(0,len(neighbors),size=1)
        return neighbors[ranind[0]]


    def get_random_video(self, index):
        # dissimilar videos have middle distance from the query video (tag is 2)
        others = np.where(self.v_feats[index]==2)    

        others=others[0]
        # If there is no tag 2 video, we get dissimilar videos with tag 0
        if len(others)==0:
            others = np.where(self.v_feats[index]==0)
            others=others[0]

        ranind = np.random.randint(0,len(others),size=1)
        return others[ranind[0]]

    def __getitem__(self, item):
        item2, is_similar = self.random_video(item)
        t2 = self.video_feats[item2]
        t1 = self.video_feats[item]
        t1_random, t1_label = self.random_frame(t1)
        t2_random, t2_label = self.random_frame(t2)
        neighbor1 = torch.from_numpy(self.achors[self.neighbor[item][0]])
        neighbor2 = torch.from_numpy(self.achors[self.neighbor[item2][0]])
        visual_word = np.concatenate((t1,t2),0)
        mask_input = np.concatenate((t1_random,t2_random),0)

        output = {"mask_input": mask_input,
                  "visual_word": visual_word,
                  "is_similar": is_similar,
                  'n1':neighbor1,
                  'n2':neighbor2}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __len__(self):
        return len(self.video_feats)

class TestDataset(data.Dataset):

    def __init__(self, feature_h5_path):
        with h5py.File(feature_h5_path,'r') as h5_file:
          self.video_feats = h5_file['feats'][:]    


    def __getitem__(self, item):

        visual_word = self.video_feats[item]

        output = {"visual_word": visual_word}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __len__(self):
        return len(self.video_feats)


def get_train_loader(feature_h5_path, sim_path,batch_size=10, shuffle=True, num_workers=1, pin_memory=True):
    v = TrainDataset(feature_h5_path,sim_path)
    data_loader = torch.utils.data.DataLoader(dataset=v,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(feature_h5_path, batch_size=256, shuffle=False, num_workers=1, pin_memory=False):
    vd = TestDataset(feature_h5_path)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              #collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader

