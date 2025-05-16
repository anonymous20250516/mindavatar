import os
import cv2
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import hcp_utils
from PIL import Image
from einops import rearrange
import random
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


clothes_map_en2idx = {
    "bodysuit": 0,
    "coat": 1,
    "dress": 2,
    "hoodie": 3,
    "jacket": 4,
    "jumpsuit": 5,
    "overcoat": 6,
    "shirt": 7,
    "suit": 8,
    "sweater": 9,
    "t-shirt": 10,
    "undershirt": 11,
    "vest": 12
}

hair_map_en2idx = {
    "buzz_cut": 0,
    "side_part": 1,
    "short_loose_hair": 2,
    "bun": 3
}

def gen_roi():
    roi_names = ['V1', 'V2', 'V3', 'V3A', 'V3B', 'V3CD', 'V4', 'LO1', 'LO2', 'LO3', 'PIT', 'V4t', 'V6', 'V6A', 'V7', 'V8', 'PH', 'FFC', 'IP0', 'MT', 
                'MST', 'FST', 'VVC', 'VMV1', 'VMV2', 'VMV3', 'PHA1', 'PHA2', 'PHA3', 'TE2p', 'IPS1']
    roi = np.zeros_like(hcp_utils.mmp.map_all).astype(np.bool_)
    rois = {}

    for i in roi_names:
        temp_name=f'L_{i}'
        for k,v in hcp_utils.mmp.labels.items():
            if v==temp_name:
                temp_idx=k
        rois[i] = np.where(((hcp_utils.mmp.map_all == temp_idx)) | (hcp_utils.mmp.map_all == (180+temp_idx)))[0] # left + 180 = right
        roi[rois[i]]=1
    print(f'{roi.sum()} voxels')
    return roi


class FMRIDataset(Dataset):
    def __init__(self, sub, split='train', norm_stats=None, use_vc=True, 
                category=None, fps = 30, train_num=0, use_time=list(range(0, 13)), subset='full'):
        super(FMRIDataset, self).__init__()
        self.sub = sub
        self.split = split

        fmri_data_path = f'./dataset/processed/sub0{sub}/sub0{sub}_fmri_run_norm.npy'
        annot_path = f'./dataset/annotation/sub0{sub}_annot.csv'
        self.video_dir = './dataset/stimuli'
        self.id_frames_dir = './dataset/stimuli/id_frames'

        self.fmri_data = np.load(fmri_data_path, allow_pickle=True)  # 假定形状为 (N, T, voxels)
        self.data_annot = pd.read_csv(annot_path)

        if subset == 'param':
            self.fmri_data = self.fmri_data[-480:]
            self.data_annot = self.data_annot[-480:]


        # Calculate male and female classes separately
        # note: some video is 60 fps
        self.data_annot['fps'] = (self.data_annot['video_end_frame'] - self.data_annot['video_start_frame']) / 8
        self.data_annot['start_time'] = self.data_annot['video_start_frame'] / self.data_annot['fps']

        male_classes = self.data_annot[self.data_annot['category'] == 'male']['start_time']
        female_classes = self.data_annot[self.data_annot['category'] == 'female']['start_time']

        # Get the number of male classes (we'll add this to female classes)
        male_class_num = max(male_classes) + 1 if not male_classes.empty else 0
        # Assign motion_class with offset for females
        self.data_annot['motion_class'] = self.data_annot.apply(
            lambda row: int(row['video_start_frame']) // fps if row['category'] == 'male' 
            else (int(row['video_start_frame']) // fps) + male_class_num,
            axis=1
        )
        # Total number of motion classes (male + female)
        self.motion_class_num = max(self.data_annot['motion_class']) + 1

        motion_cfg_path_female = f"./dataset/stimuli/motion_cfgs/female.npz"
        motion_cfg_path_male = f"./dataset/stimuli/motion_cfgs/male.npz"
        motion_cfg_female = np.load(motion_cfg_path_female)
        motion_cfg_male = np.load(motion_cfg_path_male)

        motion_cfg_female_full = motion_cfg_female["full"]
        motion_cfg_male_full = motion_cfg_male["full"]

        target_len = 10
        def extract_motion_cfg(row):
            start = int(row['start_time']) * 30
            end = start + 240 
            if row['category'] == 'female':
                cfg_seq = motion_cfg_female_full[start:end]
            else:
                cfg_seq = motion_cfg_male_full[start:end]

            indices = np.linspace(0, len(cfg_seq) - 1, target_len).astype(int)
            resampled = cfg_seq[indices]  
            return resampled.reshape(-1)  

        self.data_annot['motion_cfg'] = self.data_annot.apply(extract_motion_cfg, axis=1)
        self.motion_cfg_dim = target_len * motion_cfg_male_full.shape[-1]

        self.data_annot['PCA'] = self.data_annot['PCA'].apply(ast.literal_eval)

        valid_pca = self.data_annot[self.data_annot['PCA'] != 0]['PCA']
        if not valid_pca.empty:
            pca_dim = len(valid_pca.iloc[0])
        else:
            raise ValueError("No valid PCA vectors found to determine dimension.")

        self.data_annot['PCA'] = self.data_annot['PCA'].apply(
            lambda x: [0.0] * pca_dim if x == 0 else x
        )

        if split == 'train':
            if category is not None:
                mask = (self.data_annot['train_test_label'] == 'train') & (self.data_annot['category'] == category)
            else:
                mask = self.data_annot['train_test_label'] == 'train'
        else:
            if category is not None:
                mask = (self.data_annot['train_test_label'] == 'test') & (self.data_annot['category'] == category)
            else:
                mask = self.data_annot['train_test_label'] == 'test'
        
        self.annot = self.data_annot[mask].reset_index(drop=True)

        self.fmri = self.fmri_data[mask.values]

        if use_vc:
            roi = gen_roi()
            self.fmri = self.fmri[:, :, roi]

        self.fmri = self.fmri[:, use_time, :]
        self.fmri = self.fmri.reshape(self.fmri.shape[0], -1)
        self.fmri = np.nan_to_num(self.fmri, nan=0.0)

        self.category = self.annot["category"].values
        self.pca = np.vstack(self.annot["PCA"].values).astype(np.float32)
        self.pca_dim = self.pca.shape[-1]
        self.voxel_num = self.fmri.shape[-1]
        
        if train_num > 0:
            train_num = min(train_num, self.fmri.shape[0])
            self.fmri = self.fmri[:train_num]
        print('Date Length: ', self.__len__())

        self.cloth_class_num = 13
        self.hair_class_num = 4
        self.cloth_detail_class_num = 47

    def __len__(self):
        return self.fmri.shape[0]

    def __getitem__(self, idx):
        fmri_sample = self.fmri[idx]
        pca_sample = self.pca[idx]

        fmri_tensor = torch.from_numpy(fmri_sample).float()
        pca_tensor = torch.from_numpy(pca_sample).float()

        category_sample = self.category[idx]
        motion_class = self.annot['motion_class'][idx]
        motion_cfg = self.annot['motion_cfg'][idx]
        motion_cfg = torch.from_numpy(motion_cfg).float()

        cloth_sample = self.annot["Clothes_Up"][idx]
        cloth_idx = clothes_map_en2idx.get(cloth_sample, -1)
        cloth_detail_idx = self.annot["cloth_detail_class"][idx]

        hair_sample = self.annot["hair"][idx]
        hair_idx = hair_map_en2idx.get(hair_sample, -1) if hair_sample else -1

        texture_idx = int(self.annot["dif"][idx])
        
        # canonical portrait path
        video_name = self.annot["video_name"][idx]
        image_path = os.path.join(self.id_frames_dir, category_sample, video_name + ".jpg")
        
        
        return {'fmri': fmri_tensor, 
                'pca': pca_tensor, 
                'category': category_sample,
                'motion_class': motion_class,
                'motion_cfg': motion_cfg, 
                'cloth_idx': cloth_idx, 
                'hair_idx': hair_idx, 
                'image_path': image_path,
                'video_name': video_name,
                'texture_id': texture_idx,
                'cloth_detail_idx': cloth_detail_idx,
                }

