#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析模型在测试集上的推荐结果的类别分布
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import faiss
import signal
import traceback
import argparse

# 定义错误标志和信号处理函数
error_flag = {'sig': 0}

def sig_handler(signum, frame):
    error_flag['sig'] = signum
    print("segfault core", signum)

signal.signal(signal.SIGSEGV, sig_handler)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_parser, to_tensor, get_DataLoader, get_model, load_model, get_exp_name, load_item_cate, compute_diversity

def load_item_categories(dataset_name="tmall"):
    """
    加载物品类别信息
    参数:
        dataset_name: 数据集名称
    返回:
        item_category_map: 物品类别映射字典，键为物品ID，值为类别ID
    """
    print(f"Loading item categories from {dataset_name} dataset...")
    
    # 尝试从文件中加载类别信息
    item_cate_file = f"./data/{dataset_name}_data/{dataset_name}_item_cate.txt"
    
    item_category_map = {}
    
    try:
        print(f"Trying to load item category mapping from {item_cate_file}...")
        with open(item_cate_file, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                if len(conts) == 2:
                    item_idx, cate_id = int(conts[0]), int(conts[1])
                    item_category_map[item_idx] = str(cate_id)  # 转为字符串以保持一致性
        print(f"Successfully loaded categories for {len(item_category_map)} items")
    except FileNotFoundError:
        print(f"Warning: Could not find {item_cate_file}, unable to load item categories")
        # 如果需要后续操作，可以在这里添加备选方案
    
    return item_category_map

def get_model_recommendations(model, test_data, hidden_size, device, topN=20, args=None):
    """
    使用模型获取推荐结果
    参数:
        model: 加载的推荐模型
        test_data: 测试数据加载器
        hidden_size: 隐藏层大小
        device: 计算设备
        topN: 推荐物品数量
        args: 其他参数
    返回:
        user_recommendations: 用户推荐结果字典，键为用户ID，值为推荐物品ID列表
    """
    print(f"Generating model recommendations, topN={topN}...")
    
    user_recommendations = {}
    model.eval()
    
    # 初始化faiss索引
    gpu_indexs = [None]
    
    for i in range(1000):
        try:
            item_embs = model.output_items().cpu().detach().numpy()
            res = faiss.StandardGpuResources()  # 使用单个GPU
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = device.index
            gpu_indexs[0] = faiss.GpuIndexFlatIP(res, hidden_size, flat_config)  # 建立GPU index用于Inner Product近邻搜索
            gpu_indexs[0].add(item_embs) # 给index添加向量数据
            if error_flag['sig'] == 0:
                break
            else:
                print("Core received", error_flag['sig'])
                error_flag['sig'] = 0
        except Exception as e:
            print("Error received", e)
            traceback.print_tb(e.__traceback__)
        print("Faiss re-try", i)
    
    user_count = 0
    
    # 获取每个用户的推荐结果
    for batch_idx, (users, targets, items, mask, times) in enumerate(test_data):
        # 获取用户嵌入
        time_mat, adj_mat = times
        time_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))
        
        user_embs, item_att_w = model(to_tensor(items, device), to_tensor(users, device), None, to_tensor(mask, device), time_tensor, device, train=False)
        user_embs = user_embs.cpu().detach().numpy()
        gpu_index = gpu_indexs[0]
        
        # 根据模型输出的维度决定检索方式
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            for i, user_id in enumerate(users):
                user_recommendations[int(user_id)] = I[i].tolist()
        else:  # 多兴趣模型评估
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])  # shape=(batch_size*num_interest, embedding_dim)
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            
            for i, user_id in enumerate(users):
                item_list_set = set()
                
                # 将多兴趣模型的推荐结果合并并按相似度排序
                item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1), np.repeat(range(ni), topN)))
                item_list.sort(key=lambda x:x[1], reverse=True)  # 降序排序，内积越大，向量越近
                
                for j in range(len(item_list)):  # 按距离由近到远遍历推荐物品列表
                    item_id, score, interest_id = item_list[j]
                    if item_id not in item_list_set and item_id != 0:
                        item_list_set.add(item_id)
                        if len(item_list_set) >= topN:
                            break
                
                user_recommendations[int(user_id)] = list(item_list_set)
        
        user_count += len(users)
        if batch_idx % 10 == 0:
            print(f"Processed {user_count} users...")
    
    print(f"Completed model recommendations for {len(user_recommendations)} users")
    return user_recommendations

def analyze_category_distribution(recommendations, item_category_map):
    """
    分析推荐结果中类别出现的频次分布
    参数:
        recommendations: 用户推荐结果字典，键为用户ID，值为推荐物品ID列表 (映射后)
        item_category_map: 物品类别映射字典，键为物品ID，值为类别ID 
    返回: 
        frequency_counts: 字典，键为出现次数，值为出现该次数的类别数量
    """
    print("Analyzing category frequency distribution of recommendations...")
    
    # 统计每个类别在所有用户推荐结果中出现的总次数
    frequency_counts = defaultdict(int)

    for user_id, item_list in recommendations.items():
        category_counts = defaultdict(int)
        for item_id in item_list:
            if item_id in item_category_map:
                category = item_category_map[item_id]
                category_counts[category] += 1
    
        # 统计不同出现次数的类别数量
        for cat, count in category_counts.items():
            frequency_counts[count] += 1
    
    return dict(frequency_counts)

def plot_category_distribution(counts, title, filename):
    """
    绘制类别频次分布图
    """
    print(f"Plotting distribution: {title}")
    
    # 排序数据
    sorted_counts = sorted(counts.items(), key=lambda x: x[0])
    frequencies = [x[0] for x in sorted_counts]  # 出现次数
    num_categories = [x[1] for x in sorted_counts]  # 类别数量
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制频次计数
    plt.bar(frequencies, num_categories, alpha=0.7)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Number of Categories")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Chart saved to: {filename}")



def main():
    """主函数"""
    # 获取命令行参数
    print(sys.argv)
    parser = get_parser()
    args = parser.parse_args()
    
    print(f"Analyzing {args.model_type} model recommendations on {args.dataset} dataset...")
    
    # 路径设置
    data_path = f"./data/{args.dataset}_data/"
    output_dir = f"./analysis/{args.dataset}_{args.model_type}_imbalance_results/"
    test_file = data_path + f"{args.dataset}_test.txt"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载物品类别映射
    item_category_map = load_item_categories(args.dataset)
    
    # 使用get_exp_name函数动态生成实验名称，与evalution.py中的test函数保持一致
    batch_size = 256
    lr = 0.001
    hidden_size = 64
    seq_len = 100
    interest_num = 4
    topN = 20
    exp_name = get_exp_name(args.dataset, args.model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=False)
    model_path = "./best_model/" + exp_name + "/"
    
    print(f"Using model: {exp_name}")
    print(f"Model path: {model_path}")
    
    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 获取物品和用户数量
    if args.dataset == 'tmall':
        item_count = 946102 + 1
        user_count = 438379 + 1
    elif args.dataset == 'taobao':
        item_count = 1708531 + 1
        user_count = 976779 + 1
    
    print(f"Item count: {item_count}, User count: {user_count}")
    
    # 加载模型
    model = get_model(args.dataset, args.model_type, item_count, user_count, batch_size, hidden_size, interest_num, seq_len, device=device, args=args)
    load_model(model, model_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {args.model_type}")
    
    # 准备测试数据加载器
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0, args=args)
    
    # 使用模型生成推荐结果
    user_recommendations = get_model_recommendations(model, test_data, hidden_size, device, topN, args=args)
    
    # 保存推荐结果
    with open(output_dir + "model_recommendations.json", "w") as f:
        # 将整数键转换为字符串
        serializable_recs = {str(k): [int(item) for item in v] for k, v in user_recommendations.items()}
        json.dump(serializable_recs, f)
    print("Recommendation results saved")
    
    # 分析推荐结果的类别分布
    category_counts = analyze_category_distribution(user_recommendations, item_category_map)
    
    # 保存分布数据
    with open(output_dir + "recommendation_category_counts.json", "w") as f:
        json.dump(category_counts, f)
    
    # 绘制分布图
    plot_category_distribution(
        category_counts,
        f"{args.dataset}-{args.model_type} Recommendation Category Distribution",
        output_dir + "recommendation_category_distribution.png"
    )
    
    print("Analysis completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()