import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel, CapsuleNetwork

class PAMI(BasicModel):
    def __init__(self, item_num, user_num, hidden_size, batch_size, interest_num=4, seq_len=50, add_pos=True, args=None, device=None):
        super(PAMI, self).__init__(item_num, user_num, hidden_size, batch_size, seq_len)
        self.interest_num = interest_num
        self.num_heads = interest_num
        self.hard_readout = True
        self.add_pos = add_pos
        
        # 位置编码
        if self.add_pos:
            self.position_embedding = nn.Parameter(torch.Tensor(1, self.seq_len, self.hidden_size))
            
        # 物品和用户的转换层
        self.item_hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
            nn.Tanh()
        )
        self.user_hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
            nn.Tanh()
        )

        
        # 物品注意力层
        self.item_att_layer = nn.Linear(self.hidden_size * 4, self.num_heads, bias=False)
        
        self.alpha = args.alpha if args is not None else 0.5
        self.reset_parameters()

    def forward(self, item_list, user_list, label_list, mask, times, device, train=True):
        item_eb = self.item_embeddings(item_list)    # 物品嵌入
        user_eb = self.user_embeddings(user_list)    # 用户嵌入
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            label_eb = self.item_embeddings(label_list)
            
        # 历史物品嵌入序列
        item_eb = torch.reshape(item_eb, (-1, self.seq_len, self.hidden_size))
        
        # 添加位置编码
        if self.add_pos:
            item_eb_add_pos = item_eb + self.position_embedding.repeat(item_eb.shape[0], 1, 1)
        else:
            item_eb_add_pos = item_eb
        
        # 计算物品和用户的隐层表示
        item_hidden = self.item_hidden_layer(item_eb_add_pos)  # [batch_size, seq_len, hidden_size*4]
#        user_hidden = self.user_hidden_layer(user_eb)  # [batch_size, hidden_size*4]
#        print(user_eb.shape)
        user_hidden = user_eb.repeat(1, 4)  # [batch_size, hidden_size*4]
#        print(user_hidden.shape)

        # 计算用户注意力权重
        user_att_w = torch.matmul(item_hidden, user_hidden.unsqueeze(2))  # [batch_size, seq_len, 1]
        user_att_w = user_att_w.transpose(2,1)  # [batch_size, 1, seq_len]
        user_att_w = user_att_w.repeat(1, self.num_heads, 1)  # [batch_size, num_heads, seq_len]
        
        # 计算物品注意力权重
        item_att_w = self.item_att_layer(item_hidden)  # [batch_size, seq_len, num_heads]
        item_att_w = item_att_w.transpose(2,1)  # [batch_size, num_heads, seq_len]
        
        # 注意力掩码处理
        atten_mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)
        
        item_att_w = torch.where(atten_mask == 0, paddings, item_att_w)
        user_att_w = torch.where(atten_mask == 0, paddings, user_att_w)

        # Softmax
        item_att_w = F.softmax(item_att_w, dim=-1)
        user_att_w = F.softmax(user_att_w, dim=-1)
        
        # 融合两种注意力
        att_w = (1 - self.alpha) * item_att_w + self.alpha * user_att_w
        
        # 计算用户兴趣表示
        interest_emb = torch.matmul(att_w, item_eb)  # [batch_size, num_heads, hidden_size]
        user_eb = interest_emb
        
        if not train:
            return user_eb, att_w
            
        readout, selection = self.read_out(user_eb, label_eb)
        scores = None if self.is_sampler else self.calculate_score(readout)
        
        return user_eb, scores, att_w, readout, selection