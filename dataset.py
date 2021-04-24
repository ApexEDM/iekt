import tqdm
import os
import pickle
import logging as log
import torch
from torch.utils import data
import math
import random

class Dataset(data.Dataset):
    def __init__(self, problem_number, concept_num, root_dir, split='train'):
        super().__init__()
        self.map_dim = 0
        self.prob_encode_dim = 0
        self.path = root_dir
        self.problem_number = problem_number
        self.concept_num = concept_num
        self.show_len = 100
        self.split = split
        self.data_list = []
        log.info('Processing data...')
        self.process()
        log.info('Processing data done!')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def collate(self, batch):
        seq_num, y  = [], [] 
        x = []
        seq_length = len(batch[0][1][1]) 
        x_len = len(batch[0][1][0][0])
        for i in range(0, seq_length):
            this_x = []
            for j in range(0, x_len):
                this_x.append([])
            x.append(this_x)
        for data in batch:
            this_seq_num, [this_x, this_y] = data
            seq_num.append(this_seq_num)
            for i in range(0, seq_length):
                for j in range(0, x_len):
                    x[i][j].append(this_x[i][j])
            y += this_y[0 : this_seq_num]
        batch_x, batch_y =[], []
        for i in range(0, seq_length):
            x_info = []
            for j in range(0, x_len):
                
                if j == 2 or j == 6:    
                    x_info.append(torch.tensor(x[i][j]))
                else:
                    x_info.append(torch.tensor(x[i][j]).float())
            batch_x.append(x_info)
        return [torch.tensor(seq_num), batch_x], torch.tensor(y).float()

    def get_user_emb(self, related_concept_index, original_user_emb):         
        this_user_emb = original_user_emb.copy()
        for concept in related_concept_index:
            if concept == 0:
                continue 
            this_user_emb[concept] += 1
        
        return  this_user_emb


    def get_prob_emb(self, problem_id):
        pro_id_bin = bin(problem_id).replace('0b', '')
        prob_ini_emb = []
        for pro_id_bin_i in pro_id_bin:
            appd = 0
            if pro_id_bin_i == '1':
                appd = 1
            prob_ini_emb.append(appd)
            
        while len(prob_ini_emb) < self.prob_encode_dim:
            prob_ini_emb = [0] + prob_ini_emb
        return prob_ini_emb
    
    def get_skill_emb(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def get_related_mat(self, skills):
        skill_mat = []
        for i in skills:
            this_sk = [0] * self.concept_num
            if i != 0: 
                this_sk[i] = 1
            skill_mat.append(this_sk)
        return skill_mat

    def get_after(self, after):
        rt_after = [0] * (self.concept_num - 1)
        for i in after:
            rt_after[i - 1] = 1
        return rt_after

    def data_reader(self, stu_records):
        '''
        @params:
            stu_record: learning history of a user
        @returns:
            
        '''
        x_list = []
        y_list = []
        last_show = [300] * self.concept_num
        show_count = [0] * self.concept_num
        pre_response = 0

        for i in range(0, len(stu_records)):
            
            order_id, problem_id,skills, response= stu_records[i]
            prob = self.get_skill_emb(skills) 
            prob_bin = self.get_prob_emb(problem_id)
       
            operate = [1]
            if response == 0:
                operate = [0] #避免除0报错
            real_concepts_num = 0
            zero_filter = []
            last_show_emb = [0] * self.show_len
            show_count_emb = [0] * self.show_len
            for s in skills:
                if s != 0:
                    real_concepts_num += 1
                    zero_filter.append(1)
                    if last_show[s] < self.show_len:
                        last_show_emb[last_show[s]] = 1
                    if show_count[s] != 0:
                        if show_count[s] < self.show_len:
                            show_count_emb[show_count[s]] = 1
                        else:
                            show_count_emb[self.show_len - 1] = 1 
                else:
                    zero_filter.append(0)
            if real_concepts_num == 0:
                real_concepts_num = 1 #避免除0报错
          
            
            related_concept_matrix = None

            if len(skills) < 5:    
                related_concept_matrix = self.get_related_mat( skills)
            else:
                related_concept_matrix = 0
            
            x_list.append([
                last_show_emb,
                prob,
                skills,
                show_count_emb,
                operate,
                zero_filter,
                problem_id,
                related_concept_matrix
                ])
            y_list.append(torch.tensor(response))
            
            for si in range(0, self.concept_num):
                if si != 0 and si in skills:
                    show_count[si] += 1
                    last_show[si] = 1
                elif last_show[si] != 300:
                    last_show[si] += last_show[s]

        return x_list, y_list

    def process(self):

        self.prob_encode_dim = int(math.log(self.problem_number,2)) + 1
        with open(self.path + 'history_' + self.split + '.pkl', 'rb') as fp:
            histories = pickle.load(fp)
        loader_len = len(histories.keys())
        log.info('loader length: {:d}'.format(loader_len))
        proc_count = 0
        for k in tqdm.tqdm(histories.keys()):

            stu_record = histories[k]
            if stu_record[0] < 10:
                continue
            dt = self.data_reader(stu_record[1])
            
            self.data_list.append([stu_record[0], dt])
            proc_count += 1
        log.info('final length {:d}'.format(len(self.data_list)))




