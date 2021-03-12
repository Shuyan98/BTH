import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super(SegmentEmbedding,self).__init__(3, embed_size, padding_idx=0)
