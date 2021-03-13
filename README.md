# BTH
This repo holds the Pytorch codes and models for the BTH framework presented on CVPR 2021
**Self-supervised Video Hashing via Bidirectional Transformers**
Shuyan Li, Xiu Li, Jiwen Lu, Jie Zhou

[//]: ------------------------------Separator------------------------------

# Usage Guide

## Environment
Pytorch 0.4.1

## Data Preparation

### Download Features

VGG features are kindly uploaded by the authors of [SSVH]. You can download them from Baiduyun disk.

FCV: https://pan.baidu.com/s/1i65ccHv and YFCC: https://pan.baidu.com/s/1bqR8VCF  

Please set the data_root and home_root in args.py. 
You can place these features to in data_root.


### Preprocess

These following data should be prepared before training. Some of them for FCVID have been provided:
1. Latent features \bar{\bm{h}}. We have uploaded them in ./data/latent_feats.h5. You can also generate this file by yourself.
You should first train BTH model with only mask_loss, and use save_nf function in eval.py to generate it. 
2. Anchor set. We have uploaded it in ./data/anchors.h5. You can also generate this file by running get_anchors.py.
3. Pseudo labels. We have uploaded them in ./data/train_assit.h5. You can also generate this file by running prepare.py.
4. Similarity matrix. You can directly run apro_adj.py to generate sim_matrix.h5 in data_root. Since this file is very large, we didn't upload it.

## Training BTH
After correctly setting the path, you can run train.py to train the model. Models will be saved in ./models. 

## Testing BTH
When training is done, you can run eval.py to test it. mAP files will be save in ./results.
We also provide a model trained on FCVID for testing: ./models/fcv_bits_64/9288.pth.

## Citation

Please cite the following paper if you feel BTH useful to your research

```
@inproceedings{BTH2021CVPR,
  author    = {Shuyan Li and
               Xiu Li and
               Jiwen Lu and
               Jie Zhou},
  title     = {Self-supervised Video Hashing via Bidirectional Transformers},
  booktitle = {CVPR},
  year      = {2021},
}
```
## Contact
For any question, please file an issue or contact Lily:
email: lishuyan_lily@hotmail.com

[SSVH]:https://github.com/lixiangpengcs/Self-Supervised-Video-Hashing
