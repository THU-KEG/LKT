import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from gat_layer import GraphAttentionLayer
import numpy as np
from gat import GAT, SpGAT
import scipy.sparse as sp
from gcn import GCN
from attention import ScaledDotProductAttention
import torch.nn.functional as F
from multiattention import MultiHeadAttention


def padding_mask(seq_k, seq_q):
	# seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(-1)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, hidden_dim, video_n, dataset_path, ran, emb, device):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = exer_n + video_n + knowledge_n
        self.stu_dim = self.knowledge_dim
        self.hidden_dim = hidden_dim
        self.video_n = video_n
        self.student_n = student_n
        self.bert_dim = 768

        self.prednet_input_len = 123 * 1
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable
        
        self.dense_input_len = 101

        # self.one = torch.ones([1,16])
        # self.one = self.one.to(device)
        self.ran = torch.zeros([1,16])
        # self.ran = self.ran.to(device)

        super(Net, self).__init__()

        # network structure
        # self.emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        graph_path = 'data/course_graph_assist09.npy'
        # graph_path = os.path.join(dataset_path, 'course_graph_last2.npy')
        # graph_path = os.path.join(dataset_path, 'course_graph_new_novideo.npy')
        graph = np.load(graph_path)
        self.graph = torch.FloatTensor(graph).to(device)
        # g = sp.csr_matrix(graph)
        

        # emb = emb[:519]
        # self.emb = emb.to(device)
        
        # self.attention = ScaledDotProductAttention()
        # self.attention = MultiHeadAttention(model_dim=16, num_heads=1)

        # self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.student_emb = nn.Embedding(self.student_n, 64)

        # self.emb = torch.eye(self.exer_n).to(device)
        # self.cuxin = nn.Embedding(self.student_n, 1)
        # self.yunqi = nn.Embedding(self.student_n, 1)

        # self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        # self.video_info = nn.Embedding(self.video_n, self.knowledge_dim)
        self.concept_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        # self.concept_emb2 = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.emb = nn.Embedding(self.exer_n, 64)

        '''exercise2concept = np.load('exercise2concept.npy',allow_pickle=True).item()
        exdata = np.zeros((self.exer_n,self.knowledge_dim))
        self.emb = torch.from_numpy((exdata))
        self.emb = self.emb.float()
        for key, value in exercise2concept.items():
            l = len(value)
            for i in value:
                # print(torch.LongTensor([key]),torch.LongTensor([i]))
                self.emb[torch.LongTensor([key])] += self.concept_emb1(torch.LongTensor([i])) + self.e_discrimination(torch.LongTensor([key])) * self.concept_emb2(torch.LongTensor([i]))
                # exdata[key] += self.concept_emb1(torch.LongTensor(i)) + self.e_discrimination(torch.LongTensor(key)) * self.concept_emb2(torch.LongTensor(i))
            self.emb[torch.LongTensor([key])] /= l
            # exdata[key] /= l
        self.emb = self.emb.to(device)'''
        # self.emb = torch.from_numpy((exdata))

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.6)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.6)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        

        # self.dense_full = nn.Linear(self.dense_input_len, 16)

        self.gc = GCN(nfeat=self.exer_n, 
                nhid=100, 
                nclass=123, 
                dropout=0.6)
        # self.gc = GAT(nfeat=768, nhid=self.hidden_dim, nclass=16, dropout=0.1, alpha=0.3, nheads=8)
        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb, device, video_id, kn_ids):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        
        # emb_id = torch.LongTensor([i for i in range(self.exer_n)]).to(device)
        # emb = torch.sigmoid(self.gc(self.emb, self.graph))
        # emb = torch.sigmoid(self.gc(self.emb(emb_id), self.graph))
        # # emb = torch.cat((emb, self.ran), 0)
        
        exer_emb = emb[exer_id]
        # e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))

        stu_emb = self.student_emb(stu_id)


        # video_emb = emb[video_id]
        # print(kn_emb.size())

        # a = exer_id.reshape((exer_id.size(0),1))
        # input = torch.cat((a, video_id), 1)
        
        # dense_output = torch.sigmoid(self.dense_full(stu_emb*kn_emb))
        # attention_input = torch.cat((dense_output.unsqueeze(1), e_discrimination.unsqueeze(1)),1)
        # attention_input = torch.cat((attention_input,exer_emb.unsqueeze(1)),1)
        # attention_input = torch.cat((attention_input, video_emb), 1)
        # attention_out, _ = self.attention(attention_input, attention_input, attention_input)
        # input_x = attention_out.reshape(stu_emb.size(0),-1)


        # attention_input = torch.cat(((stu_emb*kn_emb).unsqueeze(1), e_discrimination.unsqueeze(1)),1)
        # attention_input = torch.cat((attention_input, exer_emb.unsqueeze(1)), 1)
        # # attention_input = torch.cat((attention_input, video_emb), 1)
        # attention_out, _ = self.attention(attention_input, attention_input, attention_input, None)
        # # print(attention_out.size())
        # input_x = attention_out.view(stu_emb.size(0),-1)
        # print(input_x.size())
        # attention_input = torch.cat((exer_emb.reshape((exer_emb.size(0),1,exer_emb.size(1))), video_emb), 1)
        # attention_mask = padding_mask(input, input)
        # print(attention_mask[0])
        # attention_outout, _ = self.attention(attention_input, attention_input, attention_input, 1/8, None)
        # print(attention_outout.size())
       
        # kn_emb_kuo = kn_emb.unsqueeze(1).repeat(1,16)
        # input_x = torch.cat(((stu_emb-exer_emb)*kn_emb, e_discrimination), 1)
        # input_x = torch.cat((input_x, e_discrimination), 1)
        # input_x = torch.cat((input_x, exer_emb*kn_emb), 1)
        # input_x = torch.cat((input_x, video_emb), 1)
        # print(input_x.size())
        input_x = (stu_emb - exer_emb) * kn_emb * (1- cuxin) * yunqi * 100
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    
    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data
    def get_exer_params(self, ex_id):
        emb = torch.sigmoid(self.gc(self.emb, self.graph))
        ex_emb = emb[ex_id]
        return ex_emb.data

        
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.pow((x - y), 2)
