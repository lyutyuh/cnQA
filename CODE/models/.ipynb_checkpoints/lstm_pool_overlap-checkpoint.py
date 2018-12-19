import torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json
import numpy as np
import pickle
from tqdm import tqdm
from loadcorpus import load_corpus, load_vocab
from layers.Highway import Highway


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pos_embedding_dim, que_len, ans_len,
                 vocab_size, label_size, batch_size, PADDINGIDX, posLookup, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.pos_embedding_dim = pos_embedding_dim
        self.que_len = que_len
        self.ans_len = ans_len
        self.posLookup = posLookup
        self.PADDINGIDX = PADDINGIDX

        self.pos_embeddings = nn.Embedding(len(posLookup), pos_embedding_dim, padding_idx=0)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDINGIDX)
        
        self.lstm = nn.LSTM(embedding_dim+pos_embedding_dim, hidden_dim, bidirectional=True)
        
        self.que_max_pooling = nn.MaxPool1d(que_len)
        self.ans_max_pooling = nn.MaxPool1d(ans_len)
        
        self.attention_word = nn.Linear(hidden_dim*4, hidden_dim)
        self.tanh = nn.Tanh()
        self.attention_score = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        
        self.highway = Highway(4*hidden_dim, num_layers=1, f=nn.ReLU())
        
        self.fullyconnected = nn.Linear(10*hidden_dim+2, 100)
        self.hidden2label_1 = nn.Linear(100, 20)
        self.hidden2label_2 = nn.Linear(20, label_size)
        
        self.hidden2label = nn.Sequential(self.hidden2label_1, self.hidden2label_2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        return (h0, c0)

    
    def forward(self, lq, que, q_pos, q_ovl, la, ans, a_pos, a_ovl):   
        lq, la = torch.squeeze(lq), torch.squeeze(la)  
        q_ovl, a_ovl = torch.squeeze(q_ovl), torch.squeeze(a_ovl)
        emb_que = self.word_embeddings(que)
        emb_ans = self.word_embeddings(ans)
        pos_que = self.pos_embeddings(q_pos)
        pos_ans = self.pos_embeddings(a_pos)
        
        pos_que = pos_que.view(self.que_len, self.batch_size, -1)
        emb_que = emb_que.view(self.que_len, self.batch_size, -1)
        pos_ans = pos_ans.view(self.ans_len, self.batch_size, -1)
        emb_ans = emb_ans.view(self.ans_len, self.batch_size, -1)
        
        emb_que = torch.cat([emb_que, pos_que], dim=-1)
        emb_ans = torch.cat([emb_ans, pos_ans], dim=-1)
        
        vec_que, _ = self.lstm(emb_que, self.init_hidden())
        vec_ans, _ = self.lstm(emb_ans, self.init_hidden())
        
        mask_que = torch.arange(self.que_len).expand(self.batch_size, self.que_len).cuda() < lq.float().unsqueeze(1)
        mask_ans = torch.arange(self.ans_len).expand(self.batch_size, self.ans_len).cuda() < la.float().unsqueeze(1)
        
        vec_que = vec_que.view(-1, self.batch_size, self.que_len)
        
        final_que = self.que_max_pooling(vec_que)
        final_que = torch.squeeze(final_que)
        final_que = final_que.view(self.batch_size, -1)
        
        
        vec_ans = vec_ans.view(self.ans_len, self.batch_size, -1)
        
        att_scores = []
        ans_att_vec = []
        for ih, h in enumerate(vec_ans):
            wr_expr = torch.cat([final_que, h], dim=1)
            wr_expr = self.highway(wr_expr)
            att_vec = self.tanh(self.attention_word(wr_expr))
            att_score = self.attention_score(att_vec)
            att_scores.append(att_score)
            
        att_scores = torch.stack(att_scores) # batch_size, ans_len
        att_scores = self.softmax(att_scores) # batch_size, ans_len
        
        ans_att_vec = (vec_ans*att_scores)
            
        ans_att_vec = ans_att_vec.view(-1, self.batch_size, self.ans_len)
        final_ans = self.ans_max_pooling(ans_att_vec)
        final_ans = torch.squeeze(final_ans)
        final_ans = final_ans.view(self.batch_size, -1)
        
        diff_vec = torch.abs(final_ans - final_que)
        ans_coverance = torch.sum(a_ovl, dim=1).float() / (1 + la.float())
        que_coverance = torch.sum(q_ovl, dim=1).float() / (1 + lq.float()) # normalized coverance
        
        ans_coverance = torch.unsqueeze(ans_coverance, dim=1)
        que_coverance = torch.unsqueeze(que_coverance, dim=1)
        
        
        vec_ans_covered = a_ovl.float().unsqueeze(2) * vec_ans.view(self.batch_size, self.ans_len, -1)
        vec_que_covered = q_ovl.float().unsqueeze(2) * vec_que.view(self.batch_size, self.que_len, -1)
        
        vec_que_covered = vec_que_covered.view(-1, self.batch_size, self.que_len)
        vec_ans_covered = vec_ans_covered.view(-1, self.batch_size, self.ans_len)
        vec_que_covered = self.que_max_pooling(vec_que_covered)
        vec_ans_covered = self.ans_max_pooling(vec_ans_covered)
        vec_que_covered = torch.squeeze(vec_que_covered).view(self.batch_size, -1)
        vec_ans_covered = torch.squeeze(vec_ans_covered).view(self.batch_size, -1)
        
        
        cos_sim = nn.CosineSimilarity(dim=-1)(final_que, final_ans)
        cos_sim = cos_sim.view(-1, 1)
        
        features = torch.cat([diff_vec, final_que, final_ans,\
                              vec_que_covered, vec_ans_covered,\
                              ans_coverance, que_coverance], dim=-1) # no sim
        denser_features = self.fullyconnected(features)
        # y  = cos_sim
        y = self.hidden2label(denser_features)
        return y, cos_sim

