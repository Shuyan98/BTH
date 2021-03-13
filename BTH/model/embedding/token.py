import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, frame_size, embed_size=512):
        super(TokenEmbedding,self).__init__(frame_size, embed_size, padding_idx=0)
