from __future__ import division

import sys
import logging

import torch
from torch.autograd import Variable

import torch.nn as nn
import mtdataloader
import metrics
import mtvocabulary

logger = logging.getLogger()

def train_minibatch(input, target, score, model, optimizer, criterion):
	batch_size = len(input)
	optimizer.zero_grad()
	
	input = Variable(torch.LongTensor(input))
	target = Variable(torch.LongTensor(target))
	golden = Variable(torch.FloatTensor(score))

	model.train(True)
	if model.use_cuda:
		input = input.cuda()
		target = target.cuda()
		golden = golden.cuda()

	preds = model(input, target, batch_size)

	loss = criterion(preds, golden)

	loss.backward()
	optimizer.step()

	return loss.data[0]


def train(model, vocab, args):
	optimizer = torch.optim.Adam( model.parameters(), lr=args.learning_rate, weight_decay =args.weight_decay)
	criterion = nn.MSELoss()
	data_loader = mtdataloader.DataLoader(vocab, args.train_file)

	lowest_loss = 100.00
	best_corr = -100.00
	for epoch in range(args.epochs):
		logger.info("Starting epoch %d" % epoch)
		idx = 0
		for input, target, score in data_loader.get_batches(args.batch_size):
			idx += 1
			loss = train_minibatch(input, target, score, model, optimizer, criterion)

			if idx%10==0:
				sys.stdout.write('Epoch[%d], Batch: %d, train loss: %.4f\r' % (epoch, idx, loss))
				sys.stdout.flush()
		model.eval()
		result = test(model, vocab, args)
		if result > best_corr:
			best_corr = result
			torch.save(model, args.model_file)
			logger.info('Saved new model with %.4f correlation' % result)
	# return train_org, train_pred

def test(model, vocab, args):
	data_loader = mtdataloader.DataLoader(vocab, args.test_file)
	criterion = nn.MSELoss()
	loss = 0.0
	tot_size = 0
	
	test_orgs = []
	test_preds = []

	for input, target, score in data_loader.get_batches(args.batch_size, shuffle = False):
		batch_size = len(input)
		tot_size += batch_size
		input = Variable(torch.LongTensor(input))
		target = Variable(torch.LongTensor(target))
		golden = Variable(torch.FloatTensor(score))
		if args.use_cuda:
			input = input.cuda()
			target = target.cuda()
			golden = golden.cuda()

		preds = model(input,target, batch_size)
		loss += batch_size*criterion (preds, golden)

		test_orgs.extend(golden.data.cpu().numpy().tolist()) 
		test_preds.extend(preds.data.cpu().numpy().tolist())

	result = loss / tot_size
	pr, spr, kt = metrics.calc_correl(test_orgs, test_preds)

	logger.info("Loss: %.4f, pr %.3f, spr %.3f, kt %.3f"%(result, pr, spr, kt))
	# return result.data[0]
	return(pr)
