import pickle
import numpy as np
import random
from tqdm import tqdm
import time
import math


import torch
import torch.nn.functional as F
import torch.nn as nn

import HGRanking_model
import utils

# Hyper-parameters
h_size = 128
mp_round = 3
margin1 = 1.0
margin2 = 3.0
dropout = 0.2
lr = 0.001
weight_decay = 1e-2
epochs = 5000
batch_size = 1024
loss_weight = [1.0, 1.0, 1.0]
valid_every = 5
lr_step_size = 1000
model_save_path = './HGRanking_models/'
# device = torch.device('cpu') 
device = torch.device('cuda:0')

# Constants
data_LocalIndex_path = 'FishOil_RaynaudDisease_corpus_local_index'
local2global_path = 'FishOil_RaynaudDisease_dictionary_local'
global2mesh_path = 'dict_MeSH'
FreqLabel_path = "Label_try"
whole_corpus_path = 'FishOil_RaynaudDisease_corpus'
yearly_mesh_stat_path = 'dict_stat_MeSH_year'

'''
# For monitoring avg_ranks
words_ref = ['Blood Viscosity', 'Vasoconstriction', 'Epoprostenol', 'Thrombosis', 'Platelet Aggregation', 'Arteriosclerosis','Blood Platelets','Prostaglandins E']#for fish oil and raynaud disease
'''
term_a_index = 11198 # Fish Oils
term_b_index = 6248 # Raynaud Disease


# Load files
article_list_LocalIndex = pickle.load(open(data_LocalIndex_path,'rb'))
article_list_LocalIndex = [article for article in article_list_LocalIndex if len(article)>2]

local2global_dict = pickle.load(open(local2global_path,'rb'))
global2mesh_dict = pickle.load(open(global2mesh_path, 'rb'))

# Dependent files
vocab_size = len(local2global_dict)

mesh2global_dict = {global2mesh_dict[key]:key for key in global2mesh_dict}

'''
# For monitoring avg_ranks
FreqLabel = pickle.load(open(FreqLabel_path, "rb"))
for word in words_ref:
    FreqLabel[mesh2global_dict[word]] = 1
FreqLabel[term_a_index] = 0
FreqLabel[term_b_index] = 0
'''

global2local_dict = {local2global_dict[key]:key for key in local2global_dict}
a_LocalIndex = global2local_dict[term_a_index]
b_LocalIndex = global2local_dict[term_b_index]
articles_ge = []
for i in range(vocab_size):
	if (i == a_LocalIndex) | (i == b_LocalIndex):
		articles_ge.append([a_LocalIndex,a_LocalIndex])
	else:
		articles_ge.append([a_LocalIndex,a_LocalIndex,i])

# Generating group truth ranking
terms_to_be_sorted_list = [local2global_dict[i] for i in local2global_dict]
corpus_after_cutoff = (pickle.load(open(whole_corpus_path,'rb')))['post_cutoff_data']
# yearly_mesh_stat = pickle.load(open(yearly_mesh_stat_path,'rb'))
# cutoff_year = 1985
# cooccur_after_cutoff = utils.count_all_terms_in_built_corpus(corpus_after_cutoff)
sorting_index_global = utils.rank_list_according_to_corpus(terms_to_be_sorted_list, corpus_after_cutoff)
# sorting_index_normalized_global = utils.rank_list_according_to_corpus_normalized(terms_to_be_sorted_list, corpus_after_cutoff, yearly_mesh_stat, cutoff_year, after_cutoff=True)


# Useful Functions
def gen_worse_article(article, vocab_size):
	wors = article.copy()
	while 1:
		tok = random.randint(0,vocab_size-1)
		if tok not in wors:
			wors.append(tok)
			break
	return wors

def gen_worse_articles(articles, vocab_size):
	return [gen_worse_article(art, vocab_size) for art in articles]

def original_batch_gen(articles, batch_size):
	batch_num = math.ceil(len(articles)/batch_size)
	for batch_index in range(batch_num):
		yield articles[batch_index*batch_size: (batch_index+1)*batch_size]

def gen_training_batch(articles, batch_size, vocab_size):
	random.shuffle(articles)
	batch_num = math.ceil(len(articles)/batch_size)
	for batch_index in range(batch_num):
		yield articles[batch_index*batch_size: (batch_index+1)*batch_size], gen_worse_articles(articles[batch_index*batch_size: (batch_index+1)*batch_size], vocab_size)

# Functions for Evaluation
def avg_ranks(predictions, FreqLabel, local2global_dict, global2mesh_dict, words_ref):

	ranks = np.argsort([sc*(-1) for sc in predictions])
	words_global_index = [local2global_dict[rk] for rk in ranks]
	words_ranked = [global2mesh_dict[wd] for wd in words_global_index]
	indices = [words_ranked.index(wd) for wd in words_ref]

	reserve_list = []
	for ind in range(len(words_global_index)):
		if FreqLabel[words_global_index[ind]] == 1:
			reserve_list.append(global2mesh_dict[words_global_index[ind]])

	ranks_filtered = [reserve_list.index(wd) for wd in words_ref]

	return np.mean(indices), np.mean(ranks_filtered), np.mean(ranks_filtered[0:6])

# Create the model
model = HGRanking_model.HGRanking(vocab_size, h_size, mp_round, dropout, device)

loss_func = nn.MarginRankingLoss(margin1)
# loss_func = nn.TripletMarginLoss()

print(model)
print('Number of parameters: %d'%(sum(p.numel() for p in model.parameters())))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=0.8)

# training
time_start = time.time()
loss_tr = []
valid_scores = []
spearsman_correlations = []

avg_loss = []
query_predictions = []

loss_weight_device = torch.tensor(loss_weight).long().to(device)

# training loop
for epoch in range(epochs):

	losses_tr_in_epoch = []
	# training phase
	model.train()
	for batch in gen_training_batch(article_list_LocalIndex, batch_size, vocab_size):
		originals, worses = batch
		orig_preds = model(originals)
		wors_preds = model(worses)
		targets = torch.ones(len(originals)).to(device)
		loss = torch.mean(loss_weight_device[1]*loss_func(orig_preds, wors_preds, targets) + wors_preds) # margin ranking loss
		losses_tr_in_epoch.append(loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	loss_tr_tmp = np.mean(losses_tr_in_epoch)
	loss_tr.append(loss_tr_tmp)

	scheduler.step()

	if epoch%valid_every == 0:
		# validation phase
		model.eval()
		predictions = []
		for batch in original_batch_gen(articles_ge, batch_size):
			with torch.no_grad():
				pred = model(batch)
			predictions += list(pred.data.cpu().numpy())
		# predictions = [pred[0] for pred in predictions]
		# sc1, sc2, sc3 = avg_ranks(predictions, FreqLabel, local2global_dict, global2mesh_dict, words_ref)
		# valid_scores.append([sc1, sc2, sc3])

		ranking = np.argsort([sc*(-1) for sc in predictions])
		spearsman_corr = [(utils.spearsman_correlation_at_k_two_ranks((i+1)*100, ranking, sorting_index_global))[0] for i in range(15)]
		spearsman_correlations.append(spearsman_corr)

		print("Epoch {:03d} | Time: {:01f} | Loss_tr: {:04f} \n".format(epoch, time.time()-time_start, loss_tr_tmp))
		# print("r1: {:04f}| r2: {:04f}| r3: {:04f} \n".format(sc1, sc2, sc3))
		print("sc500: {:04f} | sc800: {:04f}| sc1000: {:04f}| sc1200: {:04f} \n\n".format(spearsman_corr[4], spearsman_corr[7], spearsman_corr[9], spearsman_corr[11]))
		
	if epoch%100 == 0:
		torch.save(model.state_dict(), model_save_path+"model_epoch_%s"%(str(epoch)))

pickle.dump([loss_tr,valid_scores],open(model_save_path+'training_record.pkl','wb'))

'''
# drawing losses

def smoothList(list, strippedXs=False, degree=20):
	smoothed = [0]*(len(list)-degree+1)
	for i in range(len(smoothed)):
		smoothed[i] = sum(list[i:i+degree])/float(degree)
	return list[0:degree-1+10]+smoothed[10:]

import matplotlib.pyplot as plt
plt.plot(loss_tr,label='train')
plt.plot(smoothList(loss_tr), label='smooth')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
'''
