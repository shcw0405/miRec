import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModel import BasicModel, CapsuleNetwork

class Re4(BasicModel):
    
    def __init__(self, item_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, beta=0, args=None, device=None):
        super(Re4, self).__init__(item_num, hidden_size, batch_size, seq_len, beta)
        self.interest_num = interest_num
        self.num_heads = interest_num
        self.hard_readout = True
        self.add_pos = add_pos
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
        self.linear1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.Tanh()
            )
        self.linear2 = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        self.reset_parameters()
        
        self.fc_cons = nn.Linear(self.hidden_size, self.hidden_size * self.seq_len)#construct_loss用
        self.embedding_dim=self.hidden_size
        self.W1 = torch.nn.Parameter(data=torch.randn(256, self.embedding_dim), requires_grad=True)
        self.W1_2 = torch.nn.Parameter(data=torch.randn(self.interest_num, 256), requires_grad=True)
        self.W2 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.W3_2 = torch.nn.Parameter(data=torch.randn(self.seq_len, self.embedding_dim), requires_grad=True)
        self.W5 = torch.nn.Parameter(data=torch.randn(self.embedding_dim, self.embedding_dim), requires_grad=True)

        

    def forward(self, item_list, label_list, mask, times, device, train=True):

        item_eb = self.item_embeddings(item_list)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            label_eb = self.item_embeddings(label_list)

        # 历史物品嵌入序列，shape=(batch_size, maxlen, embedding_dim)
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))

        if self.add_pos:
            # 位置嵌入堆叠一个batch，然后与历史物品嵌入相加
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb

        # shape=(batch_size, maxlen, hidden_size*4)
        item_hidden = self.linear1(item_eb_add_pos)
        # shape=(batch_size, maxlen, num_heads)
        item_att_w  = self.linear2(item_hidden)
        # shape=(batch_size, num_heads, maxlen)
        item_att_w  = torch.transpose(item_att_w, 2, 1).contiguous()

        atten_mask = torch.unsqueeze(mask, dim=1).repeat(1, self.num_heads, 1) # shape=(batch_size, num_heads, maxlen)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1) # softmax之后无限接近于0

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = F.softmax(item_att_w, dim=-1) # 矩阵A，shape=(batch_size, num_heads, maxlen)

        # interest_emb即论文中的Vu
        interest_emb = torch.matmul(item_att_w, # shape=(batch_size, num_heads, maxlen)
                                item_eb # shape=(batch_size, maxlen, embedding_dim)
                                ) # shape=(batch_size, num_heads, embedding_dim)

        # 用户多兴趣向量
        user_eb = interest_emb # shape=(batch_size, num_heads, embedding_dim)

        if not train:
            return user_eb, None

        readout, selection = self.read_out(user_eb, label_eb)
        scores = None if self.is_sampler else self.calculate_score(readout)

        return user_eb, scores, item_att_w, readout, selection

    def calculate_re4_loss(self, interests, watch_movie_embedding, proposals_weight_logits, mask, gate, positive_weight_idx):
        # 确保输入参数为Tensor
        if isinstance(mask, list):
            mask = torch.tensor(mask, device=self.device)
        if isinstance(positive_weight_idx, list):
            positive_weight_idx = torch.tensor(positive_weight_idx, device=self.device)

        # 计算Re4的总损失
        contrastive_loss = self.contrastive_loss(interests, watch_movie_embedding, proposals_weight_logits, mask, gate)
        attend_loss = self.attend_loss(interests, watch_movie_embedding, proposals_weight_logits, mask)
        construct_loss = self.construct_loss(interests, watch_movie_embedding, positive_weight_idx, mask)
        return contrastive_loss + attend_loss + construct_loss

    def contrastive_loss(self, watch_interests, watch_movie_embedding, proposals_weight, item_mask, gate, t_cont=0.1):
        # 确保输入参数为Tensor
        if isinstance(item_mask, list):
            item_mask = torch.tensor(item_mask, device=self.device)

        # 对比损失计算实现
        norm_watch_interests = F.normalize(watch_interests, p=2, dim=-1)
        norm_watch_movie_embedding = F.normalize(watch_movie_embedding, p=2, dim=-1)
        cos_sim = torch.matmul(norm_watch_interests, torch.transpose(norm_watch_movie_embedding, 1, 2))
        positive_weight_idx = (proposals_weight > gate) * 1
        mask_cos = cos_sim.masked_fill(item_mask.unsqueeze(1), -1e9)
        pos_cos = mask_cos.masked_fill(positive_weight_idx != 1, -1e9)
        cons_pos = torch.exp(pos_cos / t_cont)
        cons_neg = torch.sum(torch.exp(mask_cos / t_cont), dim=2)
        cons_div = cons_pos / cons_neg.unsqueeze(-1)
        loss_contrastive = -torch.log(cons_div)
        return torch.mean(loss_contrastive)

    def attend_loss(self, watch_interests, watch_movie_embedding, proposals_weight_logits, item_mask):
        # 确保输入参数为Tensor
        if isinstance(item_mask, list):
            item_mask = torch.tensor(item_mask, device=self.device)

        # 注意力一致性损失计算实现
        product = torch.matmul(watch_interests, torch.transpose(watch_movie_embedding, 1, 2))
        product = product.masked_fill(item_mask.unsqueeze(1), -1e9)
        re_att = torch.softmax(product, dim=2)
        att_pred = F.log_softmax(proposals_weight_logits, dim=-1)
        return -(re_att * att_pred).sum() / (re_att).sum()

    def construct_loss(self, watch_interests, watch_movie_embedding, positive_weight_idx, item_mask):
        # 确保输入参数为Tensor
        if isinstance(item_mask, list):
            item_mask = torch.tensor(item_mask, device=self.device)
        if isinstance(positive_weight_idx, list):
            positive_weight_idx = torch.tensor(positive_weight_idx, device=self.device)

        # 重构损失计算实现
        recons_item = self.fc_cons(watch_interests)
        recons_item = recons_item.reshape([-1, self.interest_num, watch_movie_embedding.size(1), watch_movie_embedding.size(2)])

        # 调整 W3 的形状，确保它与 recons_item 的维度匹配
        recons_intermediate = torch.matmul(self.W3, torch.transpose(recons_item, 2, 3))  # 注意这里调整了 transpose 的轴

        # 调整 W3_2 的形状以匹配 recons_intermediate
        recons_weight = torch.matmul(self.W3_2.unsqueeze(0), F.tanh(recons_intermediate))

        # 重新设置形状
        recons_weight = recons_weight.reshape([-1, watch_movie_embedding.size(1), watch_movie_embedding.size(1)])
        recons_weight = recons_weight.masked_fill((watch_movie_embedding == 0).reshape(-1, 1, watch_movie_embedding.size(1)), -1e9)
        recons_weight = torch.softmax(recons_weight, dim=-1)

        # 最终重构物品嵌入
        recons_item = torch.matmul(recons_weight, torch.matmul(recons_item, self.W5))
        target_emb = watch_movie_embedding.unsqueeze(1).repeat(1, self.interest_num, 1, 1)
        loss_construct = self.recons_mse_loss(recons_item, target_emb)
        loss_construct = loss_construct.masked_fill((positive_weight_idx == 0).unsqueeze(-1), 0.)
        loss_construct = loss_construct.masked_fill(item_mask.unsqueeze(-1).unsqueeze(1), 0.)
        return torch.mean(loss_construct)