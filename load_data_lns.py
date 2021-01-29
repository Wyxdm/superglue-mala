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

        feature_gap_list = []
        for i, id in enumerate(neuron_ids):
            mask = torch.Tensor(self.labels[idx] == id).cuda()
            feature_slice = self.feature[0, :, idx, :]
            feature_m = feature_slice * mask
            # print(feature_m.shape)
            feature_gap = feature_m.sum(dim=-1).sum(dim=-1) / mask.sum()

            # 将feature拓展到128维, resize方法为简单补0, 可以改为拷贝自身到128维
            feature_gap_128 = torch.zeros(128).cuda()
            feature_gap_128[:12] = feature_gap[:]

            # print(feature_gap.shape, mask.sum())
            # print(i, id, feature_gap)
            feature_gap_list.append(feature_gap_128)
        feature_gap_list = np.transpose(np.array(feature_gap_list))

        coor_list = []
        for i, id in enumerate(neuron_ids):
            y_sum = 0
            x_sum = 0
            mask = self.labels[idx] == id
            for y in range(mask.shape[1]):
                for x in range(mask.shape[0]):
                    if mask[x][y]:
                        y_sum += y
                        x_sum += x
            coor = np.array([y_sum, x_sum]) / mask.sum()
            coor_list.append(coor)
        coor_list = np.array(coor_list).reshape((1, -1, 2))

        '''对第二张图片进行处理'''
        neuron_ids, id_counts = np.unique(self.labels[idx + 1], return_counts=True)
        # print(len(neuron_ids))
        valid_ids_index = id_counts > 1000
        neuron_ids = neuron_ids[valid_ids_index]    # 有效的neuron_ids
        # print(len(neuron_ids), id_counts)

        feature_gap_list2 = []
        for i, id in enumerate(neuron_ids):
            mask = torch.Tensor(self.labels[idx + 1] == id).cuda()
            feature_slice = self.feature[0, :, idx + 1, :]
            feature_m = feature_slice * mask
            # print(feature_m.shape)
            feature_gap = feature_m.sum(dim=-1).sum(dim=-1) / mask.sum()

            # 将feature拓展到128维, resize方法为简单补0, 可以改为拷贝自身到128维
            feature_gap_128 = torch.zeros(128).cuda()
            feature_gap_128[:12] = feature_gap[:]

            # print(feature_gap.shape, mask.sum())
            # print(i, id, feature_gap)
            feature_gap_list2.append(feature_gap_128)
        feature_gap_list2 = np.transpose(np.array(feature_gap_list2))

        coor_list2 = []
        for i, id in enumerate(neuron_ids):
            y_sum = 0
            x_sum = 0
            mask = self.labels[idx + 1] == id
            for y in range(mask.shape[1]):
                for x in range(mask.shape[0]):
                    if mask[x][y]:
                        y_sum += y
                        x_sum += x
            coor = np.array([y_sum, x_sum]) / mask.sum()
            coor_list2.append(coor)
        coor_list2 = np.array(coor_list2).reshape((1, -1, 2))


        # skip this image pair if no keypoints detected in image
        if coor_list.shape[1] < 1 or coor_list2.shape[1] < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': self.raw[idx],
                'image1': self.raw[idx + 1],
                'file_name': 'None'
            } 

        scores = np.array([0.05] * coor_list.shape[1])
        scores2 = np.array([0.05] * coor_list2.shape[1])

        return{
            'keypoints0': list(coor_list),
            'keypoints1': list(coor_list2),
            'descriptors0': list(feature_gap_list),
            'descriptors1': list(feature_gap_list2),
            'scores0': list(scores),
            'scores1': list(scores2),
            'image0': self.raw[idx],
            'image1': self.raw[idx+1],
            'all_matches': 'None',
            'file_name': 'None'
        }

if __name__ == '__main__':
    dataset = NeuronDataset(data_path='/home1/lns/Dataset/sample_A_20160501.hdf',
        model_path='/home1/lns/experiment/MALA_pytorch/models_A/model-100000.ckpt')

    train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1,
                                               drop_last=True)
    for i, pred in enumerate(train_loader):
        print(pred['descriptors0'])
        print(pred['keypoints0'])
        if i == 0: break


