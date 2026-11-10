#animation_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm # 引入tqdm以显示加载进度

def tokenize_skeleton(joints, parent_indices, n_discrete_size=128):
    token_sequence = []
    num_real_joints = joints.size(0)
    for i in range(num_real_joints):
        coords = joints[i]
        coord_tokens = ((coords - (-0.5)) / 1.0 * n_discrete_size).long().clamp(0, n_discrete_size - 1)
        parent_token = parent_indices[i]
        token_sequence.extend([
            coord_tokens[0].item() + 3,
            coord_tokens[1].item() + 3,
            coord_tokens[2].item() + 3,
            parent_token.item() + 3
        ])
    return torch.tensor(token_sequence, dtype=torch.long)

class AnimationSequenceDataset(Dataset):
    def __init__(self, dataset_dir):
        """
        【全新实现】初始化时扫描所有分片，并一次性将所有样本加载到内存列表中。
        """
        self.shard_paths = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.npz')])
        if not self.shard_paths:
            raise FileNotFoundError(f"在目录 '{dataset_dir}' 中没有找到任何 .npz 文件。")
            
        self.all_samples = []
        # 【核心修改】在这里一次性加载所有数据到 self.all_samples 列表中
        print(f"正在从 {len(self.shard_paths)} 个数据分片一次性加载到内存...")
        for path in tqdm(self.shard_paths, desc="Loading data shards into RAM"):
            try:
                with np.load(path, allow_pickle=True) as data:
                    self.all_samples.extend(list(data['sequences']))
            except Exception as e:
                print(f"警告：加载分片文件 {path} 失败: {e}")

        self.total_samples = len(self.all_samples)
        if self.total_samples == 0:
            raise ValueError("未能从任何分片文件中加载任何数据样本。")
            
        print(f"加载完成，共 {self.total_samples} 个样本已缓存到内存。")
        self.consistency_targets = None

    def update_consistency_targets(self, targets):
        print(f"正在更新 {len(targets)} 个一致性损失的伪标签...")
        assert len(targets) == self.total_samples, "伪标签数量与数据集样本总数不匹配！"
        self.consistency_targets = targets
        print("✅ 伪标签更新完成。")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        【核心修改】这个方法现在变得极其高效，因为它只从内存列表中索引，无磁盘I/O。
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of range")

        # 1. 直接从内存列表中获取样本字典
        sample_dict = self.all_samples[idx]

        # 2. 从 sample_dict 提取数据并进行词元化
        keyframe_meshes_tensor = torch.from_numpy(sample_dict['keyframe_pointclouds'])
        gt_joints = torch.from_numpy(sample_dict['anchor_joints'])
        gt_parents = torch.from_numpy(sample_dict['anchor_parents'])
        token_sequence = tokenize_skeleton(gt_joints, gt_parents)
        
        bos_token = torch.tensor([0], dtype=torch.long)
        eos_token = torch.tensor([1], dtype=torch.long)
        anchor_input_ids = torch.cat([bos_token, token_sequence])
        anchor_labels = torch.cat([token_sequence, eos_token])

        item = {
            "keyframe_meshes": keyframe_meshes_tensor,
            "anchor_input_ids": anchor_input_ids,
            "anchor_labels": anchor_labels,
        }
        
        if self.consistency_targets is not None:
            item["consistency_target"] = self.consistency_targets[idx]

        return item