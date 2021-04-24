import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging as log
import numpy
import tqdm
import pickle
from utils import batch_data_to_device

def train(model, loaders, args):
    log.info("training...")
    BCELoss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    train_sigmoid = torch.nn.Sigmoid()
    show_loss = 100
    for epoch in range(args.n_epochs):
        loss_all = 0
        for step, data in enumerate(loaders['train']):
            
            with torch.no_grad():
                x, y = batch_data_to_device(data, args.device)
            model.train()
            data_len = len(x[0])
            h = torch.zeros(data_len, args.dim).to(args.device)
            p_action_list, pre_state_list, emb_action_list, op_action_list, actual_label_list, states_list, reward_list, predict_list, ground_truth_list = [], [], [], [], [], [], [], [], []

            rt_x = torch.zeros(data_len, 1, args.dim * 2).to(args.device)
            for seqi in range(0, args.seq_len):
                ques_h = torch.cat([
                    model.get_ques_representation(x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][4].size()[0]),
                    h], dim = 1)
                flip_prob_emb = model.pi_cog_func(ques_h)

                m = Categorical(flip_prob_emb)
                emb_ap = m.sample()
                emb_p = model.cog_matrix[emb_ap,:]

                h_v, v, logits, rt_x = model.obtain_v(x[1][seqi], h, rt_x, emb_p)
                prob = train_sigmoid(logits)
                out_operate_groundtruth = x[1][seqi][4]
                out_x_groundtruth = torch.cat([
                    h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                    h_v.mul((1-out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                    dim = 1)

                out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(args.device), torch.tensor(0).to(args.device)) 
                out_x_logits = torch.cat([
                    h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                    h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                    dim = 1)                
                out_x = torch.cat([out_x_groundtruth, out_x_logits], dim = 1)

                ground_truth = x[1][seqi][4].squeeze(-1)

                flip_prob_emb = model.pi_sens_func(out_x)

                m = Categorical(flip_prob_emb)
                emb_a = m.sample()
                emb = model.acq_matrix[emb_a,:]

                h = model.update_state(h, v, emb, ground_truth.unsqueeze(1))
               
                emb_action_list.append(emb_a)
                p_action_list.append(emb_ap)
                states_list.append(out_x)
                pre_state_list.append(ques_h)
                
                ground_truth_list.append(ground_truth)
                predict_list.append(logits.squeeze(1))
                this_reward = torch.where(out_operate_logits.squeeze(1).float() == ground_truth,
                             torch.tensor(1).to(args.device), 
                             torch.tensor(0).to(args.device))
                reward_list.append(this_reward)

            seq_num = x[0]
            emb_action_tensor = torch.stack(emb_action_list, dim = 1)
            p_action_tensor = torch.stack(p_action_list, dim = 1)
            state_tensor = torch.stack(states_list, dim = 1)
            pre_state_tensor = torch.stack(pre_state_list, dim = 1)
            reward_tensor = torch.stack(reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, args.seq_len)).float()
            logits_tensor = torch.stack(predict_list, dim = 1)
            ground_truth_tensor = torch.stack(ground_truth_list, dim = 1)
            loss = []
            tracat_logits = []
            tracat_ground_truth = []
            
            for i in range(0, data_len):
                this_seq_len = seq_num[i]
                this_reward_list = reward_tensor[i]
         
                this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                        torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(args.device)
                                        ], dim = 0)
                this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                        torch.zeros(1, state_tensor[i][0].size()[0]).to(args.device)
                                        ], dim = 0)

                td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
                delta_cog = td_target_cog
                delta_cog = delta_cog.detach().cpu().numpy()

                td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
                delta_sens = td_target_sens
                delta_sens = delta_sens.detach().cpu().numpy()

                advantage_lst_cog = []
                advantage = 0.0
                for delta_t in delta_cog[::-1]:
                    advantage = args.gamma * advantage + delta_t[0]
                    advantage_lst_cog.append([advantage])
                advantage_lst_cog.reverse()
                advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(args.device)
                
                pi_cog = model.pi_cog_func(this_cog_state[:-1])
                pi_a_cog = pi_cog.gather(1,p_action_tensor[i][0: this_seq_len].unsqueeze(1))

                loss_cog = -torch.log(pi_a_cog) * advantage_cog
                
                loss.append(torch.sum(loss_cog))

                advantage_lst_sens = []
                advantage = 0.0
                for delta_t in delta_sens[::-1]:
                    # advantage = args.gamma * args.beta * advantage + delta_t[0]
                    advantage = args.gamma * advantage + delta_t[0]
                    advantage_lst_sens.append([advantage])
                advantage_lst_sens.reverse()
                advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(args.device)
                
                pi_sens = model.pi_sens_func(this_sens_state[:-1])
                pi_a_sens = pi_sens.gather(1,emb_action_tensor[i][0: this_seq_len].unsqueeze(1))

                loss_sens = - torch.log(pi_a_sens) * advantage_sens
                loss.append(torch.sum(loss_sens))
                

                this_prob = logits_tensor[i][0: this_seq_len]
                this_groud_truth = ground_truth_tensor[i][0: this_seq_len]

                tracat_logits.append(this_prob)
                tracat_ground_truth.append(this_groud_truth)

            bce = BCELoss(torch.cat(tracat_logits, dim = 0), torch.cat(tracat_ground_truth, dim = 0))
           
            label_len = torch.cat(tracat_ground_truth, dim = 0).size()[0]
            loss_l = sum(loss)
            loss = args.lamb * (loss_l / label_len) +  bce

            loss_all += loss

            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            

        show_loss = loss_all / len(loaders['train'].dataset)
        acc, auc, auroc = evaluate(model, loaders['valid'], args)
        log.info('Epoch: {:03d}, Loss: {:.7f}, acc: {:.7f}, auc: {:.7f}'.format(epoch, show_loss, acc, auc))
        
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model, os.path.join(args.run_dir, 'params_%i.pt' % epoch))
   

def evaluate(model, loader, args):
    model.eval()
    eval_sigmoid = torch.nn.Sigmoid()
    y_list, prob_list, final_action = [], [], []
    
    
    for step, data in enumerate(loader):
        
        with torch.no_grad():
            x, y = batch_data_to_device(data, args.device)
        model.train()
        data_len = len(x[0])
        h = torch.zeros(data_len, args.dim).to(args.device)
        batch_probs, uni_prob_list, actual_label_list, states_list, reward_list =[], [], [], [], []
        H = None
        if 'eernna' in args.model:
            H = torch.zeros(data_len, 1, args.dim).to(args.device)
        else:
            H = torch.zeros(data_len, args.concept_num - 1, args.dim).to(args.device)
        rt_x = torch.zeros(data_len, 1, args.dim * 2).to(args.device)
        for seqi in range(0, args.seq_len):
            ques_h = torch.cat([
                    model.get_ques_representation(x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][4].size()[0]),
                    h], dim = 1)
            flip_prob_emb = model.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = model.cog_matrix[emb_ap,:]

            h_v, v, logits, rt_x = model.obtain_v(x[1][seqi], h, rt_x, emb_p)
            prob = eval_sigmoid(logits)
            out_operate_groundtruth = x[1][seqi][4]
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                dim = 1)

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(args.device), torch.tensor(0).to(args.device)) 
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim = 1)                
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim = 1)

            ground_truth = x[1][seqi][4].squeeze(-1)

            flip_prob_emb = model.pi_sens_func(out_x)
            
            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = model.acq_matrix[emb_a,:]

            h = model.update_state(h, v, emb, ground_truth.unsqueeze(1))
            uni_prob_list.append(prob.detach())

        seq_num = x[0]
        prob_tensor = torch.cat(uni_prob_list, dim = 1)
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            batch_probs.append(prob_tensor[i][0: this_seq_len])
        batch_t = torch.cat(batch_probs, dim = 0)
        prob_list.append(batch_t)
        y_list.append(y)

    y_tensor = torch.cat(y_list, dim = 0).int()
    hat_y_prob_tensor = torch.cat(prob_list, dim = 0)

    acc = accuracy_score(y_tensor.cpu().numpy(), (hat_y_prob_tensor > 0.5).int().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(y_tensor.cpu().numpy(), hat_y_prob_tensor.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    auroc = 0

    return acc, auc, auroc

