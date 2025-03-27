#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较原始测试数据和推荐结果的类别分布比例
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_distribution_data(file_path):
    """
    加载分布数据
    
    参数:
        file_path: JSON文件路径
    返回:
        distribution_dict: 分布字典
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_proportions(distribution_dict):
    """
    计算各频次对应数量占总数的比例
    
    参数:
        distribution_dict: 分布字典
    返回:
        frequencies: 频次列表(1-20)
        proportions: 对应的比例列表
    """
    # 确保所有频次(1-20)都有值，没有的设为0
    all_frequencies = {str(i): distribution_dict.get(str(i), 0) for i in range(1, 21)}
    
    # 计算总数
    total = sum(all_frequencies.values())
    
    # 计算比例
    proportions = [all_frequencies[str(i)] / total for i in range(1, 21)]
    
    # 频次列表(1-20)
    frequencies = list(range(1, 21))
    
    return frequencies, proportions

def plot_comparison(test_file, recommend_file, output_file, dataset_name, model_name):
    """
    绘制比较折线图
    
    参数:
        test_file: 测试数据分布文件路径
        recommend_file: 推荐结果分布文件路径
        output_file: 输出图片路径
        dataset_name: 数据集名称
        model_name: 模型名称
    """
    # 加载数据
    test_distribution = load_distribution_data(test_file)
    recommend_distribution = load_distribution_data(recommend_file)
    
    # 计算比例
    test_freq, test_prop = calculate_proportions(test_distribution)
    rec_freq, rec_prop = calculate_proportions(recommend_distribution)
    
    # 绘制折线图
    plt.figure(figsize=(12, 8))
    
    # 测试数据折线
    plt.plot(test_freq, test_prop, 'b-', marker='o', linewidth=2, markersize=8, label='Original Test Data')
    
    # 推荐结果折线
    plt.plot(rec_freq, rec_prop, 'r-', marker='s', linewidth=2, markersize=8, label='Model Recommendations')
    
    # 设置图表标题和标签
    plt.title(f'Category Frequency Distribution Comparison ({dataset_name}-{model_name})', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Proportion', fontsize=14)
    
    # 设置x轴刻度
    plt.xticks(range(1, 21))
    
    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Comparison chart saved to: {output_file}")

def main():
    """主函数"""
    # 设置数据集和模型名称
    dataset_name = "tmall"
    model_name = "ComiRec-SA"
    
    # 设置文件路径
    test_file = f"./analysis/{dataset_name}_imbalance_results/category_counts.json"
    recommend_file = f"./analysis/{dataset_name}_{model_name}_imbalance_results/recommendation_category_counts.json"
    output_dir = f"./analysis/{dataset_name}_{model_name}_comparison_results/"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_distribution_comparison.png")
    
    # 绘制比较图
    plot_comparison(test_file, recommend_file, output_file, dataset_name, model_name)
    
    # 生成数据比较表格
    generate_comparison_table(test_file, recommend_file, os.path.join(output_dir, f"{dataset_name}_{model_name}_distribution_comparison.csv"))

def generate_comparison_table(test_file, recommend_file, output_file):
    """
    生成数据比较表格
    
    参数:
        test_file: 测试数据分布文件路径
        recommend_file: 推荐结果分布文件路径
        output_file: 输出CSV文件路径
    """
    # 加载数据
    test_distribution = load_distribution_data(test_file)
    recommend_distribution = load_distribution_data(recommend_file)
    
    # 计算比例
    test_freq, test_prop = calculate_proportions(test_distribution)
    rec_freq, rec_prop = calculate_proportions(recommend_distribution)
    
    # 生成CSV表格
    with open(output_file, 'w') as f:
        f.write("Frequency,Test Data Count,Test Data Proportion,Recommendation Count,Recommendation Proportion\n")
        
        for i in range(1, 21):
            test_count = test_distribution.get(str(i), 0)
            rec_count = recommend_distribution.get(str(i), 0)
            
            test_total = sum(int(v) for v in test_distribution.values())
            rec_total = sum(int(v) for v in recommend_distribution.values())
            
            test_proportion = test_count / test_total if test_total > 0 else 0
            rec_proportion = rec_count / rec_total if rec_total > 0 else 0
            
            f.write(f"{i},{test_count},{test_proportion:.6f},{rec_count},{rec_proportion:.6f}\n")
    
    print(f"Comparison table saved to: {output_file}")

if __name__ == "__main__":
    main() 