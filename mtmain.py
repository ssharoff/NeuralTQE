#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script builds a model for predicting the postedit scores for a file in the format of tab-separated lines:
wie soll Riquent wirken ?       how should Riquent expected to work ?   0.400000
(taken from the WMT18 Shared Task).
Standard word embeddings are also needed.
"""
# License: GPLv3 
# Authors: Yu Yuan, Serge Sharoff, 2018


from __future__ import division
from collections import Counter
import torch
#from scipy.stats.import pearsonr, spearmanr, kendalltau

import sys
import logging

import argparse
import random
import numpy as np

import mtvocabulary
import mttrain
import mtnetworks

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main(args):
	if args.train_file:
		vocab = mtvocabulary.Vocab(args.train_file, args.src_emb, args.tgt_emb)
		model = mtnetworks.AttentionRegression(vocab, args.emb_size, args.feature_size, args.window_size, args.dropout, args.hidden_size, args.n_layers, args.attention_size,args.use_cuda)
		if args.use_cuda:
			model.cuda()
		mttrain.train(model, vocab, args)
	else:
		model = torch.load(args.model_file, map_location=lambda storage, loc: storage)
		model.eval()
		test(model,vocab, args)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--emb_size', type = int, default= 300)
	argparser.add_argument('--feature_size', type = int, default= 200)
	argparser.add_argument('--window_size', nargs = '+', type = int, default= [1,2])
	argparser.add_argument('--dropout', type=float, default=0.3)
	argparser.add_argument('--hidden_size', type = int, default=100)
	argparser.add_argument('--n_layers', type = int, default= 1)
	argparser.add_argument('--attention_size', type = int, default= 100)
	argparser.add_argument('-t','--train_file', type=str)
	argparser.add_argument('-v','--test_file', type=str)
	argparser.add_argument('--learning_rate', type = float, default= 0.001)
	argparser.add_argument('--epochs', type = int, default= 200)
	argparser.add_argument('--batch_size', type = int, default= 1024)
	argparser.add_argument('-1','--src_emb', type=str)
	argparser.add_argument('-2','--tgt_emb', type=str)
	argparser.add_argument('-o','--prediction_file', required=True, type=str)
	argparser.add_argument('-m','--model_file', required=True, type=str)
	argparser.add_argument('--weight_decay', type=float, default=1e-6)
	argparser.add_argument('--seed_num', type=int, default=42)
	argparser.add_argument('--use_cuda', type=int, default=1)
	args, extra_args = argparser.parse_known_args()
        
	loghandler = logging.FileHandler(args.model_file+'.log')
	loghandler.setLevel(logging.INFO)
	loghandler.setFormatter(logformatter)
	logger.addHandler(loghandler)

	random.seed(args.seed_num)
	torch.manual_seed(args.seed_num)
	np.random.seed(args.seed_num)

	main(args)

