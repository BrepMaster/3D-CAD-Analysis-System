# segmentation_logic.py
import os
import json
import tempfile
import torch
import numpy as np
import dgl
from occwl.io import load_step
from graph_utils import build_graph
from constants import DEFAULT_COLORS
from segmentation_model import Segmentation
from preprocessor import load_one_graph  # 添加这行导入


class SegmentationLogic:
    def __init__(self):
        self.model = None
        self.label_mapping = None
        self.label_names = ["类别 1", "类别 2"]
        self.colors = [DEFAULT_COLORS[0].copy(), DEFAULT_COLORS[1].copy()]
        self.predicted_labels = []
        self.face_count = 0
        self.label_counts = [0] * len(self.label_names)

    def load_model(self, file_path):
        """加载模型文件"""
        self.model = Segmentation.load_from_checkpoint(file_path)
        self.model.eval()
        return os.path.basename(file_path)

    def load_labels(self, file_path):
        """加载标签配置文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)

        # 处理不同格式的标签文件
        if isinstance(self.label_mapping, dict):
            if "label_names" in self.label_mapping:
                # 新格式: 包含label_names和colors
                self.label_names = self.label_mapping["label_names"]
                self.colors = [color.copy() for color in self.label_mapping["colors"]]
            else:
                # 旧格式: {"0": "Label1", "1": "Label2"}
                sorted_items = sorted(self.label_mapping.items(), key=lambda x: int(x[0]))
                self.label_names = [name for key, name in sorted_items]
                self.colors = [DEFAULT_COLORS[int(key) % len(DEFAULT_COLORS)].copy() for key, name in sorted_items]
        else:
            # 旧格式: 直接是名称列表
            self.label_names = self.label_mapping
            self.colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)].copy() for i in range(len(self.label_names))]

        # 确保颜色数量足够
        if len(self.colors) < len(self.label_names):
            needed = len(self.label_names) - len(self.colors)
            self.colors.extend(DEFAULT_COLORS[:needed])

        # 重置计数
        self.label_counts = [0] * len(self.label_names)
        return os.path.basename(file_path)

    def update_label_config(self, config):
        """更新标签配置"""
        self.label_names = config["label_names"]
        self.colors = [color.copy() for color in config["colors"]]
        self.label_counts = [0] * len(self.label_names)

        # 如果已有预测结果，需要调整标签范围
        if len(self.predicted_labels) > 0:
            max_label = len(self.colors) - 1
            if max_label >= 0:
                self.predicted_labels = np.clip(self.predicted_labels, 0, max_label)
                # 重新计算标签计数
                unique, counts = np.unique(self.predicted_labels, return_counts=True)
                self.label_counts = [0] * len(self.label_names)
                for u, c in zip(unique, counts):
                    if u < len(self.label_counts):
                        self.label_counts[u] = c

    def process_step_file(self, step_file, mode, bin_file=None):
        """处理STEP文件进行分割"""
        if mode == 2 and bin_file:
            sample = load_one_graph(bin_file)
            inputs = sample["graph"]
            inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
            inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        elif mode == 1:
            solid = load_step(step_file)[0]
            graph = build_graph(solid, 10, 10, 10)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_bin = os.path.join(temp_dir, "temp_graph.bin")
                dgl.data.utils.save_graphs(temp_bin, [graph])

                sample = load_one_graph(temp_bin)
                inputs = sample["graph"]
                inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
                inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        else:
            raise ValueError("无效的分割模式")

        with torch.no_grad():
            logits = self.model(inputs)
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            max_label = len(self.colors) - 1
            self.predicted_labels = np.clip(predicted, 0, max_label)

        # 更新统计信息
        unique, counts = np.unique(self.predicted_labels, return_counts=True)
        self.face_count = len(self.predicted_labels)
        self.label_counts = [0] * len(self.label_names)
        for u, c in zip(unique, counts):
            if u < len(self.label_counts):
                self.label_counts[u] = c

        return self.predicted_labels

    def load_seg_file(self, file_path):
        """加载SEG分割结果文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.predicted_labels = []

            for line in lines:
                line = line.strip()
                if line:
                    try:
                        label = int(line)
                        self.predicted_labels.append(label)
                    except ValueError:
                        parts = line.split()
                        for part in parts:
                            if part:
                                self.predicted_labels.append(int(part))

        if not self.predicted_labels:
            raise ValueError("SEG文件没有包含有效的标签数据")

        # 裁剪标签到有效范围
        if self.colors:
            max_label = len(self.colors) - 1
            self.predicted_labels = np.clip(self.predicted_labels, 0, max_label)

        # 更新统计信息
        unique, counts = np.unique(self.predicted_labels, return_counts=True)
        self.face_count = len(self.predicted_labels)
        self.label_counts = [0] * len(self.label_names)
        for u, c in zip(unique, counts):
            if u < len(self.label_counts):
                self.label_counts[u] = c

        return os.path.basename(file_path)

    def get_label_info(self):
        """获取标签信息"""
        return {
            "names": self.label_names,
            "colors": self.colors,
            "counts": self.label_counts,
            "total_faces": self.face_count
        }

    def get_predicted_labels(self):
        """获取预测标签"""
        return self.predicted_labels.copy()

    def reset(self):
        """重置所有状态"""
        self.model = None
        self.label_mapping = None
        self.label_names = ["类别 1", "类别 2"]
        self.colors = [DEFAULT_COLORS[0].copy(), DEFAULT_COLORS[1].copy()]
        self.predicted_labels = []
        self.face_count = 0
        self.label_counts = [0] * len(self.label_names)