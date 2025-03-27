#!/usr/bin/env python
# -*- coding: utf-8 -*-
#完整流程了 很好的代码 大脑旋转
"""
提取测试集用于模型输入的序列，并分析类别分布
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_parser, to_tensor, get_DataLoader, get_model, load_model

def load_test_data(test_file, user_map_file, item_map_file, dataset_name="tmall"):
    """
    加载测试数据
    参数:
        test_file: 测试文件路径
        user_map_file: 用户映射文件路径
        item_map_file: 物品映射文件路径
        dataset_name: 数据集名称，默认为"tmall"
    返回: 用户历史行为字典
        return user_history, user_map, item_map
        user_history: 字典，键为用户原始ID，值为用户原始历史行为列表
        user_map[映射后id] = 原始id
        item_map[映射后id] = 原始id
    """
    print(f"加载{dataset_name}测试数据...")

    # 加载用户和物品映射
    user_map = {}
    with open(user_map_file, 'r') as f:
        for line in f:
            user, idx = line.strip().split(',')
            user_map[int(idx)] = int(user)     #user_map[映射后id] = 原始id
    
    item_map = {}
    with open(item_map_file, 'r') as f:
        for line in f:
            item, idx = line.strip().split(',')
            item_map[int(idx)] = int(item)
    
    # 加载测试数据
    user_history = {}
    current_user = None
    current_items = []
    
    with open(test_file, 'r') as f:
        for line in f:
            user_idx, item_idx, pos, timestamp = line.strip().split(',')
            user_idx, item_idx = int(user_idx), int(item_idx)
            
            if current_user is None:
                current_user = user_idx
            
            if user_idx != current_user:
                user_history[current_user] = current_items
                current_user = user_idx
                current_items = []
            
            current_items.append(item_idx)
    
    # 不要忘记最后一个用户
    if current_user is not None:
        user_history[current_user] = current_items
    
    print(f"已加载 {len(user_history)} 个用户的测试数据")
    return user_history, user_map, item_map

def extract_item_categories(user_history, item_map, dataset_name="tmall"):
    """
    提取物品类别。尝试从{dataset}_item_cate.txt和{dataset}_cate_map.txt文件中读取真实的类别信息
    如果文件不存在，则回退到使用物品ID的第一位数字作为伪类别
    
    参数:
        user_history: 字典，键为用户原始ID，值为物品原始id列表
        item_map: 物品映射字典
        dataset_name: 数据集名称，默认为"tmall"
        item_map[映射后id] = 原始id
    返回值：
        user_category_history[用户原始id] = [映射后cateid列表]
        item_cate[映射后item_id]=映射后cate_id
    """
    print(f"提取{dataset_name}数据集的物品类别...")
    
    # 尝试从文件中加载真实类别信息
    item_cate_file = f"./data/{dataset_name}_data/{dataset_name}_item_cate.txt"
    cate_map_file = f"./data/{dataset_name}_data/{dataset_name}_cate_map.txt"
    
    item_category_map = {}
    '''
cate_map[原始cate_id]=映射后cate_id
item_cate[映射后item_id]=映射后cate_id
    '''  
    # 尝试加载物品-类别映射
    try:
        print(f"尝试从{item_cate_file}加载物品类别映射...")
        with open(item_cate_file, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                if len(conts) == 2:
                    item_idx, cate_id = int(conts[0]), int(conts[1])
                    item_category_map[item_idx] = str(cate_id)  # 转为字符串以保持一致性
        print(f"成功加载了{len(item_category_map)}个物品的类别信息")
    except FileNotFoundError:
        print(f"警告: 找不到{item_cate_file}文件，无法加载物品类别信息")

    
    # 将用户历史物品映射到类别  
    # user_history: 字典，键为用户原始ID，值为物品原始id列表
    # item_category_map[映射后item_id]=映射后cate_id
    # 需要先将原始item_id转换为映射后的item_id再查询类别
    user_category_history = {}
    # 创建原始ID到映射ID的反向映射
    item_reverse_map = {v: k for k, v in item_map.items()}  # item_reverse_map[原始id] = 映射后id
    
    for user_idx, items in user_history.items():
        # 先将原始item_id转换为映射后的item_id，再查询类别
        user_category_history[user_idx] = [item_category_map.get(item_reverse_map.get(item, -1), "unknown") for item in items]
    return user_category_history, item_category_map
    # user_category_history[用户原始id] = [映射后cateid列表]

def analyze_category_distribution(category_sequences):
    """
    分析用户历史中类别出现的频次分布
    参数:
        category_sequences: 用户类别历史，格式为 user_category_history[用户原始id] = [映射后cateid列表]
    返回: 
        frequency_counts: 字典，键为出现次数，值为出现该次数的类别数量
    """
    print("分析类别频次分布...")
    
    # 统计每个类别在所有用户历史中出现的总次数
    frequency_counts = defaultdict(int)
    for user, categories in category_sequences.items():
        category_counts = defaultdict(int)
        for cat in categories[-20:]:
            category_counts[cat] += 1
    
        # 统计不同出现次数的类别数量
        for cat, count in category_counts.items():
            frequency_counts[count] += 1

    print(frequency_counts)
    
    return dict(frequency_counts)

def plot_category_distribution(counts, title, filename):
    """
    绘制类别频次分布图
    """
    print(f"绘制分布图: {title}")
    
    # 排序数据
    sorted_counts = sorted(counts.items(), key=lambda x: x[0])
    frequencies = [x[0] for x in sorted_counts]  # 出现次数
    num_categories = [x[1] for x in sorted_counts]  # 类别数量
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制频次计数
    plt.bar(frequencies, num_categories, alpha=0.7)
    plt.title("Category Frequency Distribution")
    plt.xlabel("Frequency")
    plt.ylabel("Number of Categories")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"图表已保存到: {filename}")

def main():
    """主函数"""
    # 获取命令行参数，默认为taobao数据集
    dataset_name = "tmall"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    
    print(f"正在分析{dataset_name}数据集的类别分布...")
    
    # 路径设置
    data_path = f"./data/{dataset_name}_data/"
    output_dir = f"./analysis/{dataset_name}_imbalance_results/"
    test_file = data_path + f"{dataset_name}_test.txt"
    user_map_file = data_path + f"{dataset_name}_user_map.txt"
    item_map_file = data_path + f"{dataset_name}_item_map.txt"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 加载测试数据并提取类别
    user_history, user_map, item_map = load_test_data(test_file, user_map_file, item_map_file, dataset_name)
    user_category_history, item_category_map = extract_item_categories(user_history, item_map, dataset_name)
    '''
        user_history: 字典，键为用户原始ID，值为物品原始id列表
        user_category_history[用户原始id] = [映射后cateid列表]
        item_category_map[映射后item_id]=映射后cate_id
    '''
    # 保存用户历史行为序列（用于模型输入）
    with open(output_dir + "user_history_sequences.json", "w") as f: #用户原始id映射物品原始id list
        # 将整数键转换为字符串
        serializable_history = {str(k): v for k, v in user_history.items()}
        json.dump(serializable_history, f)
    print("用户历史行为序列已保存")
    
    # 保存类别映射
    with open(output_dir + "item_category_map.json", "w") as f:
        # 将整数键转换为字符串
        serializable_map = {str(k): v for k, v in item_category_map.items()}
        json.dump(serializable_map, f)
    
    # 步骤2: 分析并绘制类别分布
    category_counts = analyze_category_distribution(user_category_history)
    
    # 保存分布数据
    with open(output_dir + "category_counts.json", "w") as f:
        json.dump(category_counts, f)
    
    # 绘制分布图
    plot_category_distribution(
        category_counts,
        f"{dataset_name}测试集类别分布",
        output_dir + "test_category_distribution.png"
    )
    

if __name__ == "__main__":
    main()