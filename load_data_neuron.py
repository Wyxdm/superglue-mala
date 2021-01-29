import numpy as np
import torch
import os
import cv2
import h5py
from unet3d import UNet3D
from collections import OrderedDict

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def resume_params(model, path):
    checkpoint = torch.load(path)
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def model_forward(raw, labels, model):
    sel_raw = (slice(0, 40), slice(0, 625), slice(0, 625))
    input = raw[sel_raw].astype(np.float32) / 255.0
    input = np.expand_dims(np.expand_dims(input, axis=0), axis=0)
    input = torch.Tensor(input).to('cuda').contiguous()
    # print(input.shape)
    pred, feature = model(input)
    # print(pred.shape, feature.shape)  # ([1, 12, 12, 407, 407])

    z_edge = input.shape[-3] - feature.shape[-3]
    x_edge = input.shape[-2] - feature.shape[-2]
    y_edge = input.shape[-1] - feature.shape[-1]
    sel_pred = (slice(z_edge//2, -z_edge//2), slice(x_edge//2, -x_edge//2), slice(y_edge//2, -y_edge//2))
    labels = labels[sel_raw][sel_pred]
    raw = raw[sel_raw][sel_pred]
    # print(labels.shape)
    assert feature.shape[-3:] == labels.shape   # feature和mask的尺寸应该一致

    return feature, labels, raw

class NeuronDataset(Dataset):

    def __init__(self, data_path, model_path):

        model = UNet3D()
        model = resume_params(model, model_path)
        model.cuda()

        with h5py.File(data_path, 'r') as f:
            raw = f['volumes/raw'][:]
            labels = f['volumes/labels/neuron_ids'][:]

        self.feature, self.labels, self.raw = model_forward(raw, labels, model)

    def __len__(self):
        return self.labels.shape[0] - 1

    def __getitem__(self, idx):

        '''对第一张图片进行处理'''
        neuron_ids, id_counts = np.unique(self.labels[idx], return_counts=True)
        # print(len(neuron_ids))
        valid_ids_index = id_counts > 1000
        neuron_ids = neuron_ids[valid_ids_index]    # 有效的neuron_ids
        # print(len(neuron_ids), id_counts)

        # 根据每个mask做gap提取特征向量
        feature_gap_list = []
        for i, id in enumerate(neuron_ids):
            mask = torch.Tensor(self.labels[idx] == id).cuda()
            feature_slice = self.feature[0, :, idx, :]
            feature_m = feature_slice * mask
            # print(feature_m.shape)
            feature_gap = feature_m.sum(dim=-1).sum(dim=-1) / mask.sum()

            # 将feature拓展到128维, resize方法为简单补0, 可以改为拷贝自身到128维
            feature_gap_128 = torch.zeros((1,128)).cuda()
            feature_gap_128[0, :12] = feature_gap[:]

            # print(feature_gap.shape, mask.sum())
            # print(i, id, feature_gap)
            if i == 0:
                feature_gap_list = feature_gap_128
            else:
                feature_gap_list = torch.cat((feature_gap_list, feature_gap_128), dim=0)
        # print(feature_gap_list.shape) # torch.Size([30, 128])
        feature_gap_list = torch.transpose(feature_gap_list, 0, 1)
        # 原load_data代码使用了转置操作(为什么??),这里也保持一致
        # print(feature_gap_list.shape)

        # 找出平均坐标
        coor_list = []
        for i, id in enumerate(neuron_ids):
            coor_array = np.where(self.labels[idx] == id)
            y = np.average(coor_array[1])
            x = np.average(coor_array[0])
            coor = np.array([y, x])
            coor_list.append(coor)
        coor_list = np.array(coor_list).reshape((1, -1, 2))

        '''对第二张图片进行处理'''
        neuron_ids2, id_counts2 = np.unique(self.labels[idx + 1], return_counts=True)
        # print(len(neuron_ids2))
        valid_ids_index = id_counts2 > 1000
        neuron_ids2 = neuron_ids2[valid_ids_index]    # 有效的neuron_ids
        # print(len(neuron_ids2), id_counts2)

        feature_gap_list2 = []
        for i, id in enumerate(neuron_ids2):
            mask = torch.Tensor(self.labels[idx + 1] == id).cuda()
            feature_slice = self.feature[0, :, idx + 1, :]
            feature_m = feature_slice * mask
            # print(feature_m.shape)
            feature_gap = feature_m.sum(dim=-1).sum(dim=-1) / mask.sum()

            # 将feature拓展到128维, resize方法为简单补0, 可以改为拷贝自身到128维
            feature_gap_128 = torch.zeros((1, 128)).cuda()
            feature_gap_128[0, :12] = feature_gap[:]

            if i == 0:
                feature_gap_list2 = feature_gap_128
            else:
                feature_gap_list2 = torch.cat((feature_gap_list2, feature_gap_128), dim=0)
        feature_gap_list2 = torch.transpose(feature_gap_list2, 0, 1)

        coor_list2 = []
        for i, id in enumerate(neuron_ids2):
            coor_array = np.where(self.labels[idx + 1] == id)
            y = np.average(coor_array[1])
            x = np.average(coor_array[0])
            coor = np.array([y, x])
            coor_list2.append(coor)
        coor_list2 = np.array(coor_list2).reshape((1, -1, 2))

        # skip this image pair if no keypoints detected in image
        if len(neuron_ids) < 1 or len(neuron_ids2) < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': self.raw[idx],
                'image1': self.raw[idx + 1],
                'file_name': str(idx)
            } 

        # 暂时忽略特征分数,取一个固定值0.05
        scores = np.array([0.05] * coor_list.shape[1])
        scores2 = np.array([0.05] * coor_list2.shape[1])

        '''计算两张图片的匹配矩阵'''
        # print(len(neuron_ids), len(neuron_ids2))
        
        matches = np.intersect1d(neuron_ids, neuron_ids2)
        match_idx = np.array([np.where(neuron_ids == id)[0][0] for id in matches])
        match_idx2 = np.array([np.where(neuron_ids2 == id)[0][0] for id in matches])

        max_value = len(neuron_ids)
        max_value2 = len(neuron_ids2)

        missing = np.setdiff1d(np.arange(max_value), match_idx)   # 返回在ar1中但不在ar2中的已排序的唯一值
        missing2 = np.setdiff1d(np.arange(max_value2), match_idx2)

        MN = np.concatenate([match_idx[np.newaxis, :], match_idx2[np.newaxis, :]])
        MN2 = np.concatenate([missing[np.newaxis, :], (max_value2) * np.ones((1, len(missing)), dtype=np.int64)])
        MN3 = np.concatenate([(max_value) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        # 这里返回数据为什么要用list呢,tensor转list,再用torch.stack转回来中间会多出一个维度
        # 如torch.Size([128, 30]) -> torch.Size([128, 1, 30])
        return{
            'keypoints0': list(coor_list),
            'keypoints1': list(coor_list2),
            'descriptors0': list(feature_gap_list),
            'descriptors1': list(feature_gap_list2),
            'scores0': list(scores),
            'scores1': list(scores2),
            'image0': self.raw[idx],
            'image1': self.raw[idx+1],
            'all_matches': list(all_matches),
            'file_name': str(idx)
        }

if __name__ == '__main__':
    dataset = NeuronDataset(data_path='/home1/lns/Dataset/sample_A_20160501.hdf',
        model_path='/home1/lns/experiment/MALA_pytorch/models_A/model-100000.ckpt')

    train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1,
                                               drop_last=True)
    for i, pred in enumerate(train_loader):
        # print(pred['descriptors0'])
        print(pred['keypoints0'])

        print(pred['all_matches'][0].shape,pred['all_matches'][1].shape)
        print(len(pred['all_matches']))
        print(pred['all_matches'])

        print(pred['keypoints0'][0].shape, pred['keypoints1'][0].shape)
        print(len(pred['descriptors0']), pred['descriptors0'][0].shape)
        print(len(pred['descriptors1']), pred['descriptors1'][0].shape)
        # print(torch.stack(pred['descriptors0']).shape)

        # print(len(pred['scores0']), pred['scores0'])
        # print(len(pred['scores1']), pred['scores1'])


        if i == 0: break


'''
torch.Size([1, 30]) torch.Size([1, 30])
2
[tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 26]]), 
         tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])]

[tensor([[[338.7082,  64.9870],
         [391.7832, 296.6621],
         [163.4719,  39.4214],
         [379.5217, 138.9277],
         [253.0356, 347.0572],
         [337.6165, 302.5403],
         [282.8585, 129.5659],
         [206.6646, 395.7788],
         [143.8161, 164.5005],
         [ 43.2697,  48.8551],
         [ 57.6932, 344.7150],
         [253.0184, 268.6858],
         [214.3693, 173.2816],
         [ 54.7645, 129.4628],
         [356.3526, 241.2599],
         [376.0599,  28.0148],
         [276.5390, 207.0053],
         [119.5054, 256.1456],
         [224.4446,   8.8850],
         [379.5657, 378.6574],
         [184.8955, 111.4810],
         [ 21.3318, 386.7715],
         [106.7556, 382.5505],
         [355.8814, 357.8001],
         [ 44.4433, 183.0890],
         [244.8353,  59.2760],
         [ 95.1324, 200.5272],
         [322.5097, 109.4182],
         [ 30.9012, 252.9603],
         [306.1788,  12.8178]]], dtype=torch.float64)]

torch.Size([1, 30, 2]) torch.Size([1, 29, 2])
128 torch.Size([1, 30])
128 torch.Size([1, 29])

30
29
'''