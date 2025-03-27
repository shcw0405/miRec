from collections import defaultdict
import math
import sys
import time
import faiss
import numpy as np
import torch
import torch.nn as nn
import os
import signal
import pickle

error_flag = {'sig':0}

def sig_handler(signum, frame):
    error_flag['sig'] = signum
    print("segfault core", signum)

signal.signal(signal.SIGSEGV, sig_handler)

from utils import get_DataLoader, get_exp_name, get_model, load_model, save_model, to_tensor, load_item_cate, compute_diversity

def evaluate_pop(model, test_data, hidden_size, device, topN=20):
    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0
    for _, (users, targets, items, mask, times) in enumerate(test_data):
        res = model.full_sort_predict(1)


        item_list = res.argsort(0, True)[:topN]
        assert len(item_list.size()) == 1
        assert item_list.size(0) == topN

        for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
            recall = 0
            dcg = 0.0
            # item_list = set(res[0])  # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
            for no, iid in enumerate(item_list):  # 对于每一个label物品
                if iid in iid_list:  # 如果该label物品在近邻搜索的结果中
                    recall += 1
                    dcg += 1.0 / math.log(no + 2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)
            total_recall += recall * 1.0 / len(iid_list)
            if recall > 0:  # recall>0当然表示有命中
                total_ndcg += dcg / idcg
                total_hitrate += 1

        total += len(targets)  # total增加每个批次的用户数量

    recall = total_recall / total  # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total  # NDCG
    hitrate = total_hitrate * 1.0 / total  # 命中率
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}

def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None, args=None):
    if model.name=="Pop":
        return evaluate_pop(model, test_data, hidden_size, device, k)
    topN = k # 评价时选取topN
    if coef is not None:
        coef = float(coef)

#    start_time = time.time()
    gpu_indexs = [None]

    for i in range(1000):
        try:
            item_embs = model.output_items().cpu().detach().numpy()
#            print(item_embs.shape) #(81636, 512)
#            print("item_embs shape:", item_embs.shape)
#            print("hidden_size:", hidden_size)
            res = faiss.StandardGpuResources()  # 使用单个GPU
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = device.index
            gpu_indexs[0] = faiss.GpuIndexFlatIP(res, hidden_size, flat_config)  # 建立GPU index用于Inner Product近邻搜索
            gpu_indexs[0].add(item_embs) # 给index添加向量数据
            if error_flag['sig'] == 0:
                break
            else:
                print("core received", error_flag['sig'])
                error_flag['sig'] = 0
        except Exception as e:
            print("error received", e)
            import traceback
            traceback.print_tb(e.__traceback__)
        print("Faiss re-try", i)

#        time.sleep(5)
#    end_time = time.time()
#    print(f"第一段执行需要{end_time - start_time}秒")

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    total_cold = 0
    total_warm = 0
    total_cold_recall = 0.0
    total_cold_ndcg = 0.0
    total_cold_hitrate = 0
    total_warm_recall = 0.0
    total_warm_ndcg = 0.0
    total_warm_hitrate = 0
#cold user：如果用户的任何一个兴趣的关注次数小于 2，则该用户被认为是 cold user。
#valid user：如果用户的所有兴趣的关注次数都大于等于 2，则该用户被认为是 valid user。
    for _, (users, targets, items, mask, times) in enumerate(test_data): # 一个batch的数据
 #       print(np.array(users).shape)  #(batch_size,)
        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        time_mat, adj_mat = times
        time_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))
#        print(to_tensor(items, device).shape)   #(batch_size,20)
        user_embs, item_att_w= model(to_tensor(items, device), to_tensor(users, device), None, to_tensor(mask, device), time_tensor, device, train=False)
        user_embs = user_embs.cpu().detach().numpy()
#        print(user_embs.shape)    #[B,num_interest,64]
        gpu_index = gpu_indexs[0] 
#        print(item_att_w.shape)# torch.Size([512, 4, 20])
#        exceed_5_count = 0  # 记录 distribution 最大值超过 topN 一半的用户数
#        exceed_7_count = 0
#        exceed_9_count = 0
#        total_users = len(targets)  # 总用户数
#        print(item_att_w)
#        time.sleep(2)
        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2: # 非多兴趣模型评估
#            print(user_embs.shape)
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                item_list = set(I[i]) # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
                for no, iid in enumerate(item_list): # 对于每一个label物品
                    if iid in iid_list: # 如果该label物品在近邻搜索的结果中
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(I[i], item_cate_map) # 两个参数分别为推荐物品列表和物品类别字典
        else: # 多兴趣模型评估

            ni = user_embs.shape[1] # num_interest
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]]) # shape=(batch_size*num_interest, embedding_dim)
#            print(user_embs.shape)
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            
            for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                warm_recall = 0
                cold_recall = 0
                warm_dcg = 0.0
                cold_dcg = 0.0

#                print(i)
#                print(iid_list)
#                time.sleep(2)
                item_list_set = set()
                interest_distribution = [0] * ni  # 初始化兴趣分布计数器，无论任何路径都要初始化
                if coef is None: # 不考虑物品多样性
                    # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1), np.repeat(range(ni), topN)))
                    item_list.sort(key=lambda x:x[1], reverse=True) # 降序排序，内积越大，向量越近
                    for j in range(len(item_list)): # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                        item_id, score, interest_id = item_list[j]
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
#                            interest_distribution[interest_id] += 1  # 增加该兴趣的选取计数
                            if len(item_list_set) >= topN:
                                break

                
                else:         # 考虑物品多样性
                    coef = float(coef)
                    # 所有兴趣向量的近邻物品集中起来按距离再次排序
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = [] # 存放（item_id, distance, item_cate）三元组，要用到物品类别，所以只存放有类别的物品
                    tmp_item_set = set() # 近邻推荐物品中有类别的物品的集合
                    for (x, y) in origin_item_list: # x为索引，y为距离
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN): # 选出topN个物品
                        max_index = 0
                        # score = distance - λ * 已选出的物品中与该物品的类别相同的物品的数量（score越大越好）
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)): # 遍历所有候选物品，每个循环找出一个score最大的item
                            # 第一次遍历必然先选出第一个物品
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score: # 当距离得分小于max_score时，后续物品得分一定小于max_score
                                break
                        item_list_set.add(item_list[max_index][0])
                        # 选出来的物品的类别对应的value加1，这里是为了尽可能选出类别不同的物品
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index) # 候选物品列表中删掉选出来的物品
                
                # 在多兴趣模型评估的 else 分支里，user_embs与item_att_w计算完成后新增:
                    
                item_att_w_np = item_att_w.cpu().detach().numpy()  # shape = [batch_size, num_interest, seq_len]
                cold_users_indices = []
                valid_users_indices = []
#                print(item_att_w_np.shape)
#                print(targets.__len__())
#                time.sleep(10)
                for i in range(len(targets)):
                    # 对第 i 个用户的 item_att_w 求 argmax，统计每个兴趣被关注的次数
#                    print(i)
                    att_max = np.argmax(item_att_w_np[i], axis=0)  # shape = [seq_len, ]，元素范围 [0, num_interest)
                    nozero = np.where(item_att_w_np[i][0] != 0)[0][-1]  # 最后一个不为0的位置 也就是有效位置
#                    print(att_max)
#                    print(item_att_w_np[i][0])
#                    print(nozero)
#                    time.sleep(5)
                    att_max_valid = att_max[:nozero+1]  # Only consider valid positions up to nozero
                    unique_interests_count = np.bincount(att_max_valid)  # Count occurrences of each interest
#                    print(att_max_valid)
#                    print(unique_interests_count)
#                    time.sleep(5)
                    # If some interests have 0 occurrences, the length might be shorter than max interest id
                    # Pad with zeros to ensure all interests are represented
                    if len(unique_interests_count) < ni:  # ni is num_interest
                        unique_interests_count = np.pad(unique_interests_count, (0, ni - len(unique_interests_count)))
                    unique_interests = set(att_max_valid)  # Keep track of unique interests for cold/valid user classification


                    if any(count < 2 for count in unique_interests_count):
                        cold_users_indices.append(i)
                    else:
                        valid_users_indices.append(i)
#                print(cold_users_indices, '\n', valid_users_indices)
#                time.sleep(5)
                # 上述if-else只是为了用不同方式计算得到最后推荐的结果item列表
                for no, iid in enumerate(item_list_set): # 对于每一个推荐的物品
#                    print(no)
#                    print(iid_list)
#                    time.sleep(5)
                    
                    if iid in iid_list: # 如果该推荐的物品在label物品列表中
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                        if no in cold_users_indices:
                            cold_recall += 1
                            cold_dcg += 1.0 / math.log(no+2, 2)
                        else:
                            warm_recall += 1
                            warm_dcg += 1.0 / math.log(no+2, 2)
                
                idcg = 0.0
                warm_idcg = 0.0
                cold_idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list) # len(iid_list)表示label数量
                
                for no in range(warm_recall):
                    warm_idcg += 1.0 / math.log(no+2, 2)
                for no in range(cold_recall):
                    cold_idcg += 1.0 / math.log(no+2, 2)
                total_cold_recall += cold_recall * 1.0 / len(iid_list)
                total_warm_recall += warm_recall * 1.0 / len(iid_list)
                
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                
                if warm_recall > 0:
                    total_warm_ndcg += warm_dcg / warm_idcg
                    total_warm_hitrate += 1
                if cold_recall > 0:
                    total_cold_ndcg += cold_dcg / cold_idcg
                    total_cold_hitrate += 1
                
                if coef is not None:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)

#        print(cold_users_indices.__len__(), valid_users_indices.__len__())

        total += len(targets) # total增加每个批次的用户数量
        total_cold += len(cold_users_indices)
        total_warm += len(valid_users_indices)
        

    recall = total_recall / total # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total # NDCG
    hitrate = total_hitrate * 1.0 / total # 命中率
    
    warm_recall = total_warm_recall / total_warm
    warm_ndcg = total_warm_ndcg / total_warm
    warm_hitrate = total_warm_hitrate * 1.0 / total_warm
    cold_recall = total_cold_recall / total_cold
    cold_ndcg = total_cold_ndcg / total_cold
    cold_hitrate = total_cold_hitrate * 1.0 / total_cold
    
    if coef is None:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'warm_recall': warm_recall, 'warm_ndcg': warm_ndcg, 'warm_hitrate': warm_hitrate, 'cold_recall': cold_recall, 'cold_ndcg': cold_ndcg, 'cold_hitrate': cold_hitrate}
#        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    diversity = total_diversity * 1.0 / total # 多样性

    cold_recall, cold_ndcg, cold_hitrate = 0.0, 0.0, 0.0
    for idx in cold_users_indices:
        recall = 0
        dcg = 0.0
        item_list = set(I[idx])
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}

torch.set_printoptions(
    precision=2,    # 精度，保留小数点后几位，默认4
    threshold=np.inf,
    edgeitems=3,
    linewidth=200,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=False  # 用科学技术法显示数据，默认True
)

def train(device, train_file, valid_file, test_file, dataset, model_type, item_count, user_count, batch_size, lr, seq_len, 
            hidden_size, interest_num, topN, max_iter, test_iter, decay_step, lr_decay, patience, exp, args):
    
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN) # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径

    # prepare data
    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=1, args=args)
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=0, args=args)

    model = get_model(dataset, model_type, item_count, user_count, batch_size, hidden_size, interest_num, seq_len, args=args,device=device)
    model = model.to(device)
    model.set_device(device)
    
    model.set_sampler(args, device=device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=lr_decay)

    trials = 0
    
    log_file = "training_log.txt"
    epoch = 1
    
    print('training begin')
    sys.stdout.flush()

    start_time = time.time()
    try:
        total_loss, total_loss_1, total_loss_2, total_loss_3, total_loss_4, total_loss_5  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iter = 0
        best_metric = 0 # 最佳指标值，在这里是最佳recall值
        #scheduler.step()
        for i, (users, targets, items, mask, times) in enumerate(train_data):
            model.train()
            iter += 1
            optimizer.zero_grad()
            pos_items = to_tensor(targets, device)
            interests, atten, readout, selection = None, None, None, None
            time_mat, adj_mat = times
            times_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))
            if model_type in ['ComiRec-SA', "REMI", 'Re4', 'PAMI']:
                interests, scores, atten, readout, selection = model(to_tensor(items, device), to_tensor(users, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type == 'ComiRec-DR':
                interests, scores, readout = model(to_tensor(items, device), to_tensor(users, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type == 'MIND':
                interests, scores, readout, selection = model(to_tensor(items, device), to_tensor(users, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type in ['GRU4Rec', 'DNN']:
                readout, scores = model(to_tensor(items, device), to_tensor(users, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type == 'Pop':
                loss = model.calculate_loss(pos_items)
            else:
#                print(interests.shape)
#                print(readout.shape)
#                print(pos_items.shape)
#                print(selection.shape)
#                print(interests.shape)
#                print(selection)
                loss = model.calculate_sampled_loss(readout, pos_items, selection, interests) if model.is_sampler else model.calculate_full_loss(loss_fn, scores, to_tensor(targets, device), interests)

            if model_type == "REMI":
                loss += args.rlambda * model.calculate_atten_loss(atten)
                
            if model_type == "Re4":
                watch_movie_embedding = model.item_embeddings(to_tensor(items, device))  # 获取观看物品的嵌入
                loss += model.calculate_re4_loss(interests, watch_movie_embedding, atten, to_tensor(mask, device).bool(), gate=0.5, positive_weight_idx=None)

            if model_type == "PAMI":
                loss += model.calculate_pami_loss(interests, atten, to_tensor(mask, device).bool(), gate=0.5, positive_weight_idx=None)

            loss.backward()
            optimizer.step()

            total_loss += loss
        
            with open(log_file, "a") as f:
                if iter%test_iter == 0:
                    model.eval()
                    f.write(f"=== Epoch {epoch} ===\n")
                    epoch = epoch + 1
                    
                    metrics = evaluate(model, valid_data, hidden_size, device, 20, args=args)
                    log_str = 'iter: %d, train loss: %.4f \n' % (iter, total_loss / test_iter) # 打印loss
                    if metrics != {}:
                        log_str += ', '.join(['valid ' + key + '@20' + ': %.6f' % value for key, value in metrics.items()])
                    print(exp_name)
    #                print(log_str)

                    model.eval()
                    metrics = evaluate(model, valid_data, hidden_size, device, 50, args=args)
    #                log_str = 'iter: %d, train loss: %.4f' % (iter, total_loss / test_iter) # 打印loss
                    if metrics != {}:
                        log_str += '\n' + ', '.join(['valid ' + key + '@50' + ': %.6f' % value for key, value in metrics.items()])
    #                print(exp_name)
                    print(log_str)
                    f.write(log_str)

                    # 保存recall最佳的模型
                    if 'recall' in metrics:
                        recall = metrics['recall']
                        if recall > best_metric:
                            best_metric = recall
                            save_model(model, best_model_path)
                            trials = 0
                        else:
                            trials += 1
                            if trials > patience: # early stopping
                                print("early stopping!")
                                break

                    # 每次test之后loss_sum置零
                    total_loss = 0.0
                    test_time = time.time()
                    print("time interval: %.4f min" % ((test_time-start_time)/60.0))
                    sys.stdout.flush()

                if iter >= max_iter * 1000: # 超过最大迭代次数，退出训练
                    break

    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')

    load_model(model, best_model_path)
    model.eval()

    # 训练结束后用valid_data测试一次
#    metrics = evaluate(model, valid_data, hidden_size, device, 20, args=args)
#    print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))
#    metrics = evaluate(model, valid_data, hidden_size, device, 50, args=args)
#    print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))
#    

# 打开文件并追加写入模式

    with open(log_file, "a") as f:
        # 标记当前是第几轮训练
        f.write(f"=== Epoch {epoch} ===\n")
        epoch = epoch + 1
        # 训练结束后用valid_data测试一次
        metrics = evaluate(model, valid_data, hidden_size, device, 20, args=args)
        f.write(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]) + "\n")
        print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

        metrics = evaluate(model, valid_data, hidden_size, device, 50, args=args)
        f.write(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]) + "\n")
        print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))
        
        # 训练结束后用test_data测试一次
        print("Test result:")
        test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0,  args=args)
        metrics = evaluate(model, test_data, hidden_size, device, 20, args=args)
        for key, value in metrics.items():
            print('test ' + key + '@20' + '=%.6f' % value)
        f.write(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]) + "\n")

        metrics = evaluate(model, test_data, hidden_size, device, 50, args=args)
        for key, value in metrics.items():
            print('test ' + key + '@50' + '=%.6f' % value)
        f.write(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]) + "\n")

        item_embeddings = model.item_embeddings.weight.cpu().detach().numpy()
        save_path = os.path.join(best_model_path, 'item_embeddings.npy')
        np.save(save_path, item_embeddings)
        print(f"Item embeddings 已保存至 {save_path}, 形状: {item_embeddings.shape}")


def test(device, test_file, cate_file, dataset, model_type, item_count, batch_size, lr, seq_len, 
            hidden_size, interest_num, topN, args, coef=None, exp='test'):
    
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=False) # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径A

    model = get_model(dataset, model_type, item_count, user_count, batch_size, hidden_size, interest_num, seq_len, args=args,device=device)
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()
        
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0, args=args)
    if coef!=None:
        item_cate_map = load_item_cate(cate_file) # 读取物品的类型
        metrics = evaluate(model, test_data, hidden_size, device, 20, coef=coef, item_cate_map=item_cate_map)
        print(', '.join(['test ' + key + '@20' + ': %.6f' % value for key, value in metrics.items()]))
    else:
        metrics = evaluate(model, test_data, hidden_size, device, 20, coef=coef, item_cate_map=None)
        print(', '.join(['test ' + key + '@20' + ': %.6f' % value for key, value in metrics.items()]))

    if coef!=None:
        item_cate_map = load_item_cate(cate_file) # 读取物品的类型
        metrics = evaluate(model, test_data, hidden_size, device, 50, coef=coef, item_cate_map=item_cate_map)
        print(', '.join(['test ' + key + '@50' + ': %.6f' % value for key, value in metrics.items()]))
    else:
        metrics = evaluate(model, test_data, hidden_size, device, 50, coef=coef, item_cate_map=None)
        print(', '.join(['test ' + key + '@50' + ': %.6f' % value for key, value in metrics.items()]))
    


def output(device, test_file, dataset, model_type, item_count, user_count, batch_size, lr, seq_len, 
          hidden_size, interest_num, topN, exp='eval'):
    
    exp_name = 'rocket_REMI_b512_lr0.001_d64_len20_in4_top20_2025-01-16 17:46:08' # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径
    
    # 用相同的命名方式保存embedding
    output_prefix = f"{dataset}_{model_type}_bs{batch_size}_lr{lr}_hs{hidden_size}_sl{seq_len}"
    if model_type in ['ComiRec-SA', 'ComiRec-DR', 'MIND']:
        output_prefix += f"_in{interest_num}"
    
    # 加载模型
    model = get_model(dataset, model_type, item_count, user_count, batch_size, hidden_size, interest_num, seq_len)
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()
    
    # 获取测试数据加载器
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)
    
    # 获取并保存物品embedding
    item_embs = model.output_items().cpu().detach().numpy()
    np.save(f'output/{output_prefix}_item_emb.npy', item_embs)
    
    # 获取并保存用户embedding
    user_embs = model.output_users(test_data, device)
    with open(f'output/{output_prefix}_user_emb.pkl', 'wb') as f:
        pickle.dump(user_embs, f)
        
    print(f"Model loaded from: {best_model_path}")
    print(f"Embeddings saved to output/{output_prefix}_item_emb.npy and output/{output_prefix}_user_emb.pkl")
    item_embs = model.output_items() # 获取物品嵌入
    np.save('output/' + exp_name + '_emb.npy', item_embs) # 保存物品嵌入
    
    # 添加TSNE可视化代码
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # 执行TSNE降维
    tsne = TSNE(n_components=2, random_state=42)
    item_embs_2d = tsne.fit_transform(item_embs)
    
    # 绘制散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(item_embs_2d[:, 0], item_embs_2d[:, 1], alpha=0.5)
    plt.title(f'Item Embeddings Visualization ({model_type})')
    
    # 保存图片
    plt.savefig(f'output/{exp_name}_tsne.png')
    plt.close()
