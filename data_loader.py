import json
import torch
import pickle
import numpy 

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self):
        self.batch_size = 32
        self.ptr = 0
        self.data = []
        self.pre = {}
        self.post = {}
        self.new_embedding = {}


        data_file = 'data/new_lkt2/train_set_last.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, ex_n, knowledge_n, vi_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.ex_n = int(ex_n)
        self.vi_n = int(vi_n)
        

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, input_video_ids, input_video_embs, input_video_sims, input_exer_cos = [], [], [], [], [], [], [], []
        input_video_lengths = []
        input_concept_embs = []
        input_video_concepts = []
        true_cnt = 0
        false_cnt = 0
        # for count in range(self.batch_size):  
        for count in range(self.batch_size):   
            log = self.data[self.ptr + count]
            y = log['score']
            if y == 1.0:
                true_cnt += 1
            else:
                false_cnt += 1
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code] = 1.0
            video_ids = []
           
                
            if log['video'] != {}:
                input_video_ids.append(log['video']['id'])
            else:
                input_video_ids.append(self.vi_n)

            
            input_stu_ids.append(log['user_id'])
            input_exer_ids.append(log['problem_id'])
            
            input_knowledge_embs.append(knowledge_emb)
           
            ys.append(y)
       
        self.ptr += self.batch_size
        
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.FloatTensor(input_knowledge_embs), torch.LongTensor(ys), torch.LongTensor(input_video_ids)
    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type
        self.pre = {}
        self.post = {}
        self.new_embedding = {}
        
        if d_type == 'validation':
            data_file = 'data/new_lkt2/val_set_last.json'
        else:
            data_file = 'data/new_lkt2/test_set_last.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, ex_n, knowledge_n, vi_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)
            self.ex_n = int(ex_n)
            self.vi_n = int(vi_n)

       
        

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        true_cnt = 0
        false_cnt = 0
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, input_video_ids, input_video_embs, input_video_sims, input_exer_cos = [], [], [], [], [], [], [], []
        input_video_lengths = []
        input_concept_embs = []
        input_video_concepts = []
        for i in range(len(logs)):
            log = logs[i]
            y = log['score']
            if y == 1.0:
                true_cnt += 1
            else:
                false_cnt += 1
            input_stu_ids.append(user_id)
            input_exer_ids.append(log['problem_id'])
            knowledge_emb = [0.] * self.knowledge_dim
            # exer_co = [log['correct_score']] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code] = 1.0
            
           
            if log['video'] != {}:
                input_video_ids.append(log['video']['id'])
            else:
                input_video_ids.append(self.vi_n)
          
            input_knowledge_embs.append(knowledge_emb)
            
            ys.append(y)
        self.ptr += 1
       
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys), torch.LongTensor(input_video_ids)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
