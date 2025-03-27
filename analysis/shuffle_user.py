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

# 创建打乱用户嵌入的包装器类
class UserEmbeddingShuffler:
    def __init__(self, model, shuffle_mode='permute', noise_level=1.0, seed=None):
        """
        模型测试包装器，用于打乱用户嵌入
        
        参数：
        - model: 原始推荐模型
        - shuffle_mode: 打乱模式，可选 'permute'(置换), 'noise'(添加噪声), 'random'(随机化)
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
        
        # 保存原始前向传播函数
        self.original_forward = model.forward
        
        # 替换模型的forward方法
        def new_forward(items, user_ids, target_items=None, mask=None, time_matrix=None, device=None, train=True):
            """
            打乱用户嵌入的新forward函数
            """
            if not train:  # 只在测试时打乱
                if self.shuffle_mode == 'permute':
                    # 在batch内随机打乱用户ID
                    idx = torch.randperm(user_ids.shape[0], device=user_ids.device)
                    shuffled_user_ids = user_ids[idx]
                    return self.original_forward(items, shuffled_user_ids, target_items, mask, time_matrix, device, train)
                
                elif self.shuffle_mode == 'noise':
                    # 保存原始方法
                    if hasattr(self.model, 'output_user'):
                        original_output_user = self.model.output_user
                        
                        # 创建带噪声的版本
                        def noisy_output_user(*args, **kwargs):
                            user_emb = original_output_user(*args, **kwargs)
                            # 添加噪声
                            noise = torch.randn_like(user_emb) * self.noise_level
                            return user_emb + noise
                        
                        # 替换方法
                        self.model.output_user = noisy_output_user
                        
                        # 执行前向传播
                        result = self.original_forward(items, user_ids, target_items, mask, time_matrix, device, train)
                        
                        # 恢复原始方法
                        self.model.output_user = original_output_user
                        return result
                    else:
                        # 如果没有output_user方法，使用原始forward
                        return self.original_forward(items, user_ids, target_items, mask, time_matrix, device, train)
                
                elif self.shuffle_mode == 'random':
                    # 完全随机化用户ID
                    if hasattr(self.model, 'user_count'):
                        random_user_ids = torch.randint(0, self.model.user_count, user_ids.shape, device=user_ids.device)
                        return self.original_forward(items, random_user_ids, target_items, mask, time_matrix, device, train)
                    else:
                        # 保守处理：只在batch内打乱
                        idx = torch.randperm(user_ids.shape[0], device=user_ids.device)
                        shuffled_user_ids = user_ids[idx]
                        return self.original_forward(items, shuffled_user_ids, target_items, mask, time_matrix, device, train)
                
                elif self.shuffle_mode == 'zero':
                    # 将用户嵌入置为零向量
                    if hasattr(self.model, 'output_user'):
                        original_output_user = self.model.output_user
                        
                        def zero_output_user(*args, **kwargs):
                            user_emb = original_output_user(*args, **kwargs)
                            return torch.zeros_like(user_emb)
                        
                        self.model.output_user = zero_output_user
                        result = self.original_forward(items, user_ids, target_items, mask, time_matrix, device, train)
                        self.model.output_user = original_output_user
                        return result
                    else:
                        return self.original_forward(items, user_ids, target_items, mask, time_matrix, device, train)
            
            # 训练模式不打乱
            return self.original_forward(items, user_ids, target_items, mask, time_matrix, device, train)
        
        # 使用新的forward函数替换原始函数
        self.model.forward = new_forward
    
    def restore(self):
        """恢复原始模型的forward方法"""
        self.model.forward = self.original_forward


def shuffle_test(device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size, 
                hidden_size, interest_num, seq_len, topN, shuffle_mode, noise_level=1.0, seed=42):
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
    - shuffle_mode: 打乱模式 ('permute', 'noise', 'random', 'zero')
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
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)
    
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


def test_multiple_shuffle_modes(device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size, 
                               hidden_size, interest_num, seq_len, topN, seed=42):
    """测试多种打乱模式的性能影响"""
    
    results = {}
    
    # 测试不同的打乱模式
    for mode in ['permute', 'noise', 'random', 'zero']:
        print(f"\n=== 测试 {mode} 模式 ===")
        
        if mode == 'noise':
            # 测试不同噪声强度
            for noise_level in [0.1, 0.5, 1.0, 2.0]:
                print(f"\n--- 噪声强度: {noise_level} ---")
                result = shuffle_test(
                    device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size,
                    hidden_size, interest_num, seq_len, topN, mode, noise_level, seed
                )
                results[f"{mode}_{noise_level}"] = result
        else:
            # 测试其他模式
            result = shuffle_test(
                device, test_file, cate_file, dataset, model_type, item_count, user_count, batch_size,
                hidden_size, interest_num, seq_len, topN, mode, 1.0, seed
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
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="测试用户嵌入打乱对模型性能的影响")
    
    # 数据集和模型参数
    parser.add_argument('--dataset', type=str, default='book', help='数据集名称')
    parser.add_argument('--model_type', type=str, default='ComiRec-SA', help='模型类型')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--interest_num', type=int, default=4, help='兴趣数量')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    # 打乱相关参数
    parser.add_argument('--shuffle_mode', type=str, default='permute', 
                        choices=['permute', 'noise', 'random', 'zero', 'all'], 
                        help='打乱模式: permute(置换), noise(添加噪声), random(随机化), zero(置零), all(测试所有模式)')
    parser.add_argument('--noise_level', type=float, default=1.0, help='噪声强度')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()


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
            args.interest_num, seq_len, 20, args.random_seed
        )
    else:
        # 测试单个打乱模式
        results = shuffle_test(
            device, test_file, cate_file, args.dataset, args.model_type, 
            item_count, user_count, batch_size, args.hidden_size, 
            args.interest_num, seq_len, 20, args.shuffle_mode, 
            args.noise_level, args.random_seed
        )
    
    # 将结果保存到文件
    results_dir = "analysis/shuffle_user"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/shuffle_user_{args.dataset}_{args.model_type}_{args.shuffle_mode}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': {k: {k2: {k3: float(v3) for k3, v3 in v2.items()} 
                           for k2, v2 in v.items()} 
                       for k, v in results.items()}
        }, f, indent=2)
    
    print(f"结果已保存到: {filename}")
