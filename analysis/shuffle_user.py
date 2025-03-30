import os
import sys
import argparse
import torch
import numpy as np
import random
from collections import defaultdict

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_DataLoader, get_exp_name, get_model, load_model, to_tensor, load_item_cate
from evalution_copy import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="测试用户嵌入打乱对模型性能的影响")
    
    # 数据集和模型参数
    parser.add_argument('--dataset', type=str, default='tmall', help='数据集名称')
    parser.add_argument('--model_type', type=str, default='ComiRec-SA', help='模型类型')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--interest_num', type=int, default=4, help='兴趣数量')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    # 打乱相关参数
    parser.add_argument('--shuffle_mode', type=str, default='permute', 
                        choices=['permute', 'noise', 'random', 'zero', 'all', 'fixed'], 
                        help='打乱模式: permute(置换), noise(添加噪声), random(随机化), zero(置零), all(测试所有模式), fixed(固定模式)')
    parser.add_argument('--noise_level', type=float, default=1.0, help='噪声强度')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()

# 创建打乱用户嵌入的包装器类
class UserEmbeddingShuffler:
    def __init__(self, model, shuffle_mode='permute', noise_level=1.0, seed=None):
        """
        模型测试包装器，用于打乱用户嵌入
        
        参数：
        - model: 原始推荐模型
        - shuffle_mode: 打乱模式，可选 'permute'(置换), 'noise'(添加噪声), 'random'(随机化), 'zero'(置零), 'fixed'(固定模式)
        - noise_level: 噪声强度，当 shuffle_mode='noise' 时使用
        - seed: 随机种子，用于复现结果
        """
        self.model = model
        self.shuffle_mode = shuffle_mode
        self.noise_level = noise_level
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 检查是否有get_embeddings方法，如果有，保存原始方法并替换
        if hasattr(model, 'get_embeddings') and callable(getattr(model, 'get_embeddings')):
            self.original_get_embeddings = model.get_embeddings
            print("模型重写get_embeddings函数")
            
            # 定义新的get_embeddings方法
            def new_get_embeddings(item_list, user_list):
                """重写的get_embeddings方法，用于在测试时直接修改用户嵌入"""
                # 获取原始的物品和用户嵌入
                item_eb, user_eb = self.original_get_embeddings(item_list, user_list)
                
                # 只在评估模式下修改用户嵌入
                if not self.model.training:
#                    print("打乱用户嵌入")
                    device = user_eb.device
                    batch_size = user_eb.shape[0]
                    hidden_size = user_eb.shape[1]  # 用户嵌入维度
                    
                    if self.shuffle_mode == 'permute':
                        # 在batch内随机打乱用户嵌入
                        idx = torch.randperm(batch_size, device=device)
                        user_eb = user_eb[idx]
                    
                    elif self.shuffle_mode == 'random':
                        # 随机生成用户嵌入
                        user_eb = torch.randn_like(user_eb, device=device)
                        # 归一化到合理范围
                        user_eb = user_eb * user_eb.std() / torch.std(user_eb, dim=1, keepdim=True)
                    
                    elif self.shuffle_mode == 'noise':
                        # 在原用户嵌入基础上添加噪声
                        noise = torch.randn_like(user_eb, device=device) * self.noise_level
                        user_eb = user_eb + noise
                    
                    elif self.shuffle_mode == 'zero':
                        # 将用户嵌入全部置零
                        user_eb = torch.zeros_like(user_eb, device=device)
                    
                    elif self.shuffle_mode == 'fixed':
                        # 将所有用户嵌入设为同一个值（使用第一个用户的嵌入）
                        user_eb = user_eb[0:1].repeat(batch_size, 1)
                
                return item_eb, user_eb
            
            # 替换原始方法
            self.model.get_embeddings = new_get_embeddings
        else:
            print("警告：模型没有get_embeddings方法，无法直接修改用户嵌入")
            self.original_get_embeddings = None
    
    def restore(self):
        """恢复原始模型的方法"""
        # 恢复get_embeddings方法（如果有）
        if hasattr(self, 'original_get_embeddings') and self.original_get_embeddings is not None:
            self.model.get_embeddings = self.original_get_embeddings


def shuffle_test(device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size, 
                hidden_size, interest_num, seq_len, topN, shuffle_mode, noise_level=1.0, seed=42, args=None):
    """
    使用打乱用户嵌入的方式测试模型
    
    参数:
    - device: 计算设备
    - test_file: 测试数据文件路径
    - cate_file: 类别数据文件路径
    - dataset: 数据集名称
    - model_type: 模型类型
    - item_count: 物品总数
    - user_count: 用户总数
    - batch_size: 批处理大小
    - hidden_size: 隐藏层大小
    - interest_num: 兴趣数量
    - seq_len: 序列长度
    - topN: 推荐物品数量
    - shuffle_mode: 打乱模式 ('permute', 'noise', 'random', 'zero', 'fixed')
    - noise_level: 噪声强度
    - seed: 随机种子
    """
    # 获取模型保存路径
    exp_name = get_exp_name(dataset, model_type, batch_size, 0.001, hidden_size, seq_len, interest_num, topN, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    print(f"加载模型: {best_model_path}")
    
    # 加载模型
    model = get_model(dataset, model_type, item_count, user_count, batch_size, hidden_size, interest_num, seq_len)
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()
    
    # 应用用户嵌入打乱包装器
    print(f"使用 {shuffle_mode} 模式打乱用户嵌入")
    shuffle_wrapper = UserEmbeddingShuffler(model, shuffle_mode=shuffle_mode, noise_level=noise_level, seed=seed)
    
    # 获取测试数据
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0, args=args)
    
    # 评估模型
    metrics_20 = evaluate(model, test_data, hidden_size, device, 20)
    print(', '.join([f'打乱后 {key}@20: {value:.6f}' for key, value in metrics_20.items()]))
    
    metrics_50 = evaluate(model, test_data, hidden_size, device, 50)
    print(', '.join([f'打乱后 {key}@50: {value:.6f}' for key, value in metrics_50.items()]))
    
    # 恢复原始模型
    shuffle_wrapper.restore()
    
    # 评估原始模型 (不打乱)
    print("评估原始模型 (不打乱用户嵌入)")
    metrics_20_original = evaluate(model, test_data, hidden_size, device, 20)
    print(', '.join([f'原始 {key}@20: {value:.6f}' for key, value in metrics_20_original.items()]))
    
    metrics_50_original = evaluate(model, test_data, hidden_size, device, 50)
    print(', '.join([f'原始 {key}@50: {value:.6f}' for key, value in metrics_50_original.items()]))
    
    # 返回性能差异
    diff_metrics = {
        'recall@20': metrics_20_original['recall'] - metrics_20['recall'],
        'ndcg@20': metrics_20_original['ndcg'] - metrics_20['ndcg'],
        'hitrate@20': metrics_20_original['hitrate'] - metrics_20['hitrate'],
        'recall@50': metrics_50_original['recall'] - metrics_50['recall'],
        'ndcg@50': metrics_50_original['ndcg'] - metrics_50['ndcg'],
        'hitrate@50': metrics_50_original['hitrate'] - metrics_50['hitrate']
    }
    
    print(f"性能差异 (原始 - 打乱):")
    for key, value in diff_metrics.items():
        print(f"{key}: {value:.6f}")
    
    return {
        'shuffle': {'top20': metrics_20, 'top50': metrics_50},
        'original': {'top20': metrics_20_original, 'top50': metrics_50_original},
        'diff': diff_metrics
    }


def calculate_performance_change(original_metrics, shuffled_metrics):
    """
    计算性能变化百分比
    
    参数:
    - original_metrics: 原始模型性能指标
    - shuffled_metrics: 打乱后的性能指标
    
    返回:
    - 性能变化百分比字典
    """
    changes = {}
    for metric in original_metrics:
        if original_metrics[metric] > 0:  # 避免除以零
            relative_change = (shuffled_metrics[metric] - original_metrics[metric]) / original_metrics[metric] * 100
            changes[metric] = relative_change
    
    return changes


def print_performance_analysis(results):
    """
    打印性能分析结果
    
    参数:
    - results: 测试结果字典
    """
    print("\n=== 性能变化分析 ===")
    print(f"{'模式':<15} {'Recall@20':<15} {'NDCG@20':<15} {'HitRate@20':<15} {'平均变化':<15}")
    print("-" * 75)
    
    # 获取原始性能
    original = list(results.values())[0]['original']['top20']
    
    # 计算并打印每种模式的性能变化
    mode_avg_changes = {}
    for mode, result in results.items():
        shuffled = result['shuffle']['top20']
        changes = calculate_performance_change(original, shuffled)
        
        # 计算平均变化
        avg_change = sum(changes.values()) / len(changes) if changes else 0
        mode_avg_changes[mode] = avg_change
        
        print(f"{mode:<15} {changes.get('recall', 0):<15.2f}% {changes.get('ndcg', 0):<15.2f}% {changes.get('hitrate', 0):<15.2f}% {avg_change:<15.2f}%")
    
    # 找出影响最大的模式
    if mode_avg_changes:
        most_impactful_mode = min(mode_avg_changes.items(), key=lambda x: x[1])
        print(f"\n最显著影响的打乱模式: {most_impactful_mode[0]}，平均性能变化: {most_impactful_mode[1]:.2f}%")


def test_multiple_shuffle_modes(device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size, 
                               hidden_size, interest_num, seq_len, topN, seed=42, args=None):
    """测试多种打乱模式的性能影响"""
    
    results = {}
    
    # 测试不同的打乱模式
    for mode in ['permute', 'noise', 'random', 'zero', 'fixed']:
        print(f"\n=== 测试 {mode} 模式 ===")
        
        if mode == 'noise':
            # 测试不同噪声强度
            for noise_level in [0.1, 0.5, 1.0, 2.0]:
                print(f"\n--- 噪声强度: {noise_level} ---")
                result = shuffle_test(
                    device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size,
                    hidden_size, interest_num, seq_len, topN, mode, noise_level, seed, args
                )
                results[f"{mode}_{noise_level}"] = result
        else:
            # 测试其他模式
            result = shuffle_test(
                device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size,
                hidden_size, interest_num, seq_len, topN, mode, 1.0, seed, args
            )
            results[mode] = result
    
    # 打印总结
    print("\n=== 性能影响总结 ===")
    print(f"{'模式':<15} {'Recall@20':<10} {'NDCG@20':<10} {'HitRate@20':<10} {'Recall@50':<10} {'NDCG@50':<10} {'HitRate@50':<10}")
    print("-" * 80)
    
    # 打印原始性能
    original = list(results.values())[0]['original']
    print(f"{'原始':<15} {original['top20']['recall']:<10.6f} {original['top20']['ndcg']:<10.6f} "
          f"{original['top20']['hitrate']:<10.6f} {original['top50']['recall']:<10.6f} "
          f"{original['top50']['ndcg']:<10.6f} {original['top50']['hitrate']:<10.6f}")
    
    # 打印各打乱模式性能
    for mode, result in results.items():
        shuffle_metrics = result['shuffle']
        print(f"{mode:<15} {shuffle_metrics['top20']['recall']:<10.6f} {shuffle_metrics['top20']['ndcg']:<10.6f} "
              f"{shuffle_metrics['top20']['hitrate']:<10.6f} {shuffle_metrics['top50']['recall']:<10.6f} "
              f"{shuffle_metrics['top50']['ndcg']:<10.6f} {shuffle_metrics['top50']['hitrate']:<10.6f}")
    
    # 打印性能变化分析
    print_performance_analysis(results)
    
    return results

if __name__ == '__main__':
    args = parse_args()
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"使用 cuda:{args.gpu}" if torch.cuda.is_available() else f"使用 cpu, cuda:{args.gpu} 不可用")
    else:
        device = torch.device("cpu")
        print("使用 cpu")
    
    # 根据数据集设置参数
    if args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367982 + 1
        batch_size = 128
        seq_len = 20
        user_count = 603667 + 1
    elif args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708530 + 1
        batch_size = 256
        seq_len = 50
        user_count = 976779 + 1
    elif args.dataset == 'gowalla':
        path = './data/gowalla_data/'
        item_count = 308962 + 1 
        user_count = 77123 + 1
        batch_size = 256
        seq_len = 40
    elif args.dataset == 'tmall':
        batch_size = 256
        seq_len = 100
        test_iter = 200
        path = './data/tmall_data/'
        item_count = 946102 + 1
        user_count = 438379 + 1
    elif args.dataset == 'rocket':
        batch_size = 512
        seq_len = 20
        test_iter = 200
        path = './data/rocket_data/'
        item_count = 90148 + 1
        user_count = 70312 + 1
    else:
        raise ValueError(f"未知数据集: {args.dataset}")
    
    # 设置文件路径
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    
    # 执行测试
    if args.shuffle_mode == 'all':
        # 测试所有打乱模式
        results = test_multiple_shuffle_modes(
            device, test_file, cate_file, args.dataset, args.model_type, 
            item_count, user_count, batch_size, args.hidden_size, 
            args.interest_num, seq_len, 20, args.random_seed, args
        )
    else:
        # 测试单个打乱模式
        results = shuffle_test(
            device, test_file, cate_file, args.dataset, args.model_type, 
            item_count, user_count, batch_size, args.hidden_size, 
            args.interest_num, seq_len, 20, args.shuffle_mode, 
            args.noise_level, args.random_seed, args
        )
    
    # 将结果保存到文件
    results_dir = "analysis/shuffle_user"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    import json
    from datetime import datetime
    
    # 递归函数处理嵌套字典，将可以转换的值转为float
    def process_nested_dict(d):
        if isinstance(d, dict):
            return {k: process_nested_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [process_nested_dict(item) for item in d]
        elif isinstance(d, (int, float, bool, str)) or d is None:
            return d
        else:
            # 尝试转换为float，如果失败则转为字符串
            try:
                return float(d)
            except (TypeError, ValueError):
                return str(d)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/shuffle_user_{args.dataset}_{args.model_type}_{args.shuffle_mode}_{timestamp}.json"
    
    # 处理并保存结果
    data_to_save = {
        'args': vars(args),
        'results': {}
    }
    
    if args.shuffle_mode == 'all':
        data_to_save['results'] = process_nested_dict(results)
    else:
        data_to_save['results'] = {
            args.shuffle_mode: process_nested_dict(results)
        }
    
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"结果已保存到: {filename}")
