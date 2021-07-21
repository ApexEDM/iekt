#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.utils.data import DataLoader

import os
import pickle
import argparse
import logging as log

import models
import importlib
import train
from dataset import Dataset
import numpy as np
import random
import math


parser = argparse.ArgumentParser(description='IEKT')
parser.add_argument('--debug',          action='store_true',        help='log debug messages or not')
parser.add_argument('--run_exist',      action='store_true',        help='run dir exists ok or not')
parser.add_argument('--run_dir',        type=str,   default='run/1/', help='dir to save log and models')
parser.add_argument('--data_dir',       type=str,   default='data/new_mini_09/') #assistment2009-2010
parser.add_argument('--checkpoint_path',type=str,  default= 'none',   help='the path of checkpoint') 
parser.add_argument('--log_every',      type=int,   default=0,      help='number of steps to log loss, do not log if 0')
parser.add_argument('--eval_every',     type=int,   default=0,      help='number of steps to evaluate, only evaluate after each epoch if 0')
parser.add_argument('--save_every',     type=int,   default=50,      help='number of steps to save model')
parser.add_argument('--device',         type=int,   default=-1,      help='gpu device id, cpu if -1')
parser.add_argument('--model',          type=str,   default='iekt',   help='run model')
parser.add_argument('--n_layer',type=int,   default=1,      help='number of mlp hidden layers in decoder')
parser.add_argument('--dim',type=int,   default=64,     help='hidden size for nodes')
parser.add_argument('--n_epochs',       type=int,   default=300,   help='number of epochs to train')
parser.add_argument('--batch_size',     type=int,   default=200,      help='number of instances in a batch')
parser.add_argument('--lr',             type=float, default=1e-3,   help='learning rate')
parser.add_argument('--dropout',        type=float, default=0.0,   help='dropout') 
parser.add_argument('--seq_len',       type=int, default=200,   help='the length of the sequence') 
parser.add_argument('--gamma',        type=float, default=0.93,   help='graph_type') 
parser.add_argument('--cog_levels',        type=int, default=10,   help='the response action space for cognition estimation')
parser.add_argument('--acq_levels',        type=int, default=10,   help='the response action space for  sensitivity estimation')
parser.add_argument('--lamb',        type=float, default=40.0,   help='hyper parameter for loss')
parser.add_argument('--decay',type=float, default=1e-6,   help='hyper parameter for decay') 
args = parser.parse_args() 

# if args.debug:
#     args.run_exist = True
#     args.run_dir = 'debug'
# os.makedirs(args.run_dir, exist_ok=args.run_exist)


log.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%m/%d %I:%M:%S %p', level=log.DEBUG if args.debug else log.INFO)
log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, 'log.txt'), mode='w'))
log.info('args: %s' % str(args))
args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)

def preprocess():
    datasets = {}
    with open(args.data_dir + 'problem_skill_maxSkillOfProblem_number.pkl', 'rb') as fp:
            problem_number, concept_number, max_concept_of_problem = pickle.load(fp)

    
    setattr(args, 'max_concepts', max_concept_of_problem)
    setattr(args, 'concept_num', concept_number)
    setattr(args, 'problem_number', problem_number)
    setattr(args, 'prob_dim', int(math.log(problem_number,2)) + 1)
    
    for split in ['train', 'valid']:
        file_name = os.path.join(args.data_dir, 'dataset_%s.pkl' % split)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                datasets[split] = pickle.load(f)
            log.info('Dataset split %s loaded' % split)
        else:
            datasets[split] = Dataset(args.problem_number, args.concept_num, root_dir=args.data_dir, split=split)
            with open(file_name, 'wb') as f:
                pickle.dump(datasets[split], f)
            log.info('Dataset split %s created and dumpped' % split)

    loaders = {}
    for split in ['train', 'valid']:
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            collate_fn=datasets[split].collate,
            shuffle=True if split == 'train' else False
        )

    return loaders

if __name__ == '__main__':
    loaders = preprocess()
    Model = getattr(models, args.model)
    if args.checkpoint_path != 'none':
        model = torch.load(args.checkpoint_path, map_location = torch.device(args.device))
    else:
        model = Model(args).to(args.device)
    log.info(str(vars(args)))

    train.train(model, loaders, args)
