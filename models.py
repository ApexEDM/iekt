import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import math
import modules
from torch.autograd import Variable
import pickle
import random


class iekt(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.node_dim = args.dim
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.predictor = modules.funcs(args.n_layer, args.dim * 5, 1, args.dropout)
        self.cog_matrix = nn.Parameter(torch.randn(args.cog_levels, args.dim * 2).to(args.device), requires_grad=True) 
        self.acq_matrix = nn.Parameter(torch.randn(args.acq_levels, args.dim * 2).to(args.device), requires_grad=True)
        self.select_preemb = modules.funcs(args.n_layer, args.dim * 3, args.cog_levels, args.dropout) 
        self.checker_emb = modules.funcs(args.n_layer, args.dim * 12, args.acq_levels, args.dropout) 
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number - 1, args.dim).to(args.device), requires_grad=True)
        self.gru_h = modules.mygru(0, args.dim * 4, args.dim)
        showi0 = []
        for i in range(0, args.n_epochs):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(args.device)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num - 1, args.dim).to(args.device), requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()

    def get_ques_representation(self, prob_ids, related_concept_index, filter0, data_len):
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
            self.concept_emb],
            dim = 0).unsqueeze(0).repeat(data_len, 1, 1)
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)
        related_concepts = concepts_cat[r_index, related_concept_index,:]
        filter_sum = torch.sum(filter0, dim = 1)

        div = torch.where(filter_sum == 0, 
            torch.tensor(1.0).to(self.device), 
            filter_sum
            ).unsqueeze(1).repeat(1, self.node_dim)
        
        concept_level_rep = torch.sum(related_concepts, dim = 1) / div
        
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb], dim = 0)
        
        item_emb = prob_cat[prob_ids]

        v = torch.cat(
            [concept_level_rep,
            item_emb],
            dim = 1)
        return v


    def pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.select_preemb(x), dim = softmax_dim)

    def obtain_v(self, this_input, h, x, emb):
        
        last_show, problem, related_concept_index, show_count, operate, filter0, prob_ids, related_concept_matrix = this_input

        data_len = operate.size()[0]

        v = self.get_ques_representation(prob_ids, related_concept_index, filter0, data_len)
        predict_x = torch.cat([h, v], dim = 1)
        h_v = torch.cat([h, v], dim = 1)
        prob = self.predictor(torch.cat([
            predict_x, emb
        ], dim = 1))
        return h_v, v, prob, x

    def update_state(self, h, v, emb, operate):

        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.node_dim * 2)),
            v.mul((1 - operate).repeat(1, self.node_dim * 2))], dim = 1)
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.node_dim * 2)),
            emb.mul((operate).repeat(1, self.node_dim * 2))], dim = 1)
        inputs = v_cat + e_cat
        next_p_state = self.gru_h(inputs, h)
        return next_p_state
    

    def pi_sens_func(self, x, softmax_dim = 1):
        return F.softmax(self.checker_emb(x), dim = softmax_dim)


