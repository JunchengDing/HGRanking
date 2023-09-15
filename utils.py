import pickle
import numpy as np
from scipy.stats import spearmanr

##########################################
### functions to generate ground truth ###
##########################################

def count_single_term_in_built_corpus(corpus, term_k_index):

	# the courpus is built by selecting documents containing either one of the term before of after cutoff date.
	# the term_k is the candidate for the assocation terms.
	# the return is #(term_A, term_k) + #(term_C, term_k)

	cnt_k_cooccur = 0
	for doc in corpus:
		if term_k_index in doc:
			cnt_k_cooccur += 1

	return cnt_k_cooccur

def count_all_terms_in_built_corpus(corpus):

	# the courpus is built by selecting documents containing either one of the term before of after cutoff date.
	# the term_k is the candidate for the assocation terms.
	# the return is #(term_A, term_k) + #(term_C, term_k) of all possible term_k

	cnt_k_cooccur_dict = {}
	for doc in corpus:
		for term_k in doc:
			if term_k in cnt_k_cooccur_dict:
				cnt_k_cooccur_dict[term_k] += 1
			else:
				cnt_k_cooccur_dict[term_k] = 1

	return cnt_k_cooccur_dict

def count_single_term_in_whole_corpus(dict_stat, term_k_index, cutoff_year, after_cutoff=True):

	# the dict_stat is derived from the whole corpus, dict[term_index][year] indicating the counts of the term appearing in that year 
	# the term_k is the candidate for the assocation terms.
	# the return is #(k)

	cnt_k = 0
	term_stat = dict_stat[term_k_index]
	
	if after_cutoff:
		for year in term_stat:
			if year > cutoff_year:
				cnt_k += term_stat[year]
	else:
		for year in term_stat:
			if year <= cutoff_year:
				cnt_k += term_stat[year]

	return cnt_k

def rank_list_according_to_corpus(term_index_list, corpus):

	# given a list of MeSH terms in their indices in the dictionary and the build corpus
	# calculating their cooccurence in the with either terms and rank the terms
	# return the ranked list of the terms and their cooccurrences


	cnt_k_cooccur_dict = count_all_terms_in_built_corpus(corpus)

	term_cooccur_in_built_corpus = []
	for term_index in term_index_list:
		if term_index in cnt_k_cooccur_dict:
			term_cooccur_in_built_corpus.append(cnt_k_cooccur_dict[term_index]*(-1))
		else:
			term_cooccur_in_built_corpus.append(0)

	sorting_index = np.argsort(term_cooccur_in_built_corpus)

	#ranked_term_index = [term_index_list[k] for k in sorting_index]
	#ranked_term_counts = [term_cooccur_in_built_corpus[k]*(-1) for k in sorting_index]
	#return ranked_term_index, ranked_term_counts

	return sorting_index

def rank_list_according_to_corpus_normalized(term_index_list, corpus, dict_stat, cutoff_year, after_cutoff=True):

	# given a list of MeSH terms in their indices in the dictionary and the build corpus
	# calculating their cooccurence in the with either terms and rank the terms normalized by the occurrence in the whole corpus (in the same period)
	# return the ranked list of the terms and their cooccurrences

	cnt_k_cooccur_dict = count_all_terms_in_built_corpus(corpus)

	term_cooccur_in_built_corpus = []
	for term_index in term_index_list:
		if term_index in cnt_k_cooccur_dict:
			term_cooccur_in_built_corpus.append(cnt_k_cooccur_dict[term_index]*(-1))
		else:
			term_cooccur_in_built_corpus.append(0)


	term_count_in_whole_corpus = []
	for term_index in term_index_list:
		term_count_in_whole_corpus.append(count_single_term_in_whole_corpus(dict_stat, term_index, cutoff_year, after_cutoff=True))

	term_normalized_score = []
	for index in range(len(term_index_list)):
		if term_count_in_whole_corpus[index]:
			term_normalized_score.append(term_cooccur_in_built_corpus[index]*np.log(1/term_count_in_whole_corpus[index]))#TF-IDF style normalization
			#term_normalized_score.append(term_cooccur_in_built_corpus[index]/term_count_in_whole_corpus[index])#normal style normalization
		else:
			term_normalized_score.append(0)

	sorting_index = np.argsort(term_normalized_score)

	#ranked_term_index = [term_index_list[k] for k in sorting_index]
	#ranked_term_scores = [term_normalized_score[k]*(-1) for k in sorting_index]
	#return ranked_term_index, ranked_term_scores

	return sorting_index

###########################################
### functions for preprocessing to data ###
###########################################

def gen_corpus_specific_dict(corpus, corpus_name):

	# input the corpus as a list of lists
	# output are written in the files: dict_corpus_specific, in which is a list-structured data mapping local index into global index
	# the corpus are also formatted into local indices as corpus_flie+'_local_index'


	# first scan to generate the mappling list items
	mapping_list = []
	for doc in corpus:
		for wd in doc:
			if wd not in mapping_list:
				mapping_list.append(wd)

	# generating the dictionary and inversed dictionary
	local_dictionary = {}
	#revesed_local_dictionary = {}
	for i,wd in enumerate(mapping_list):
		local_dictionary[i] = wd
	revesed_local_dictionary = {local_dictionary[i]:i for i in local_dictionary}

	# second scan to change the corpus into the local-indexed format
	corpus_local_index = []
	for doc in corpus:
		corpus_local_index.append([revesed_local_dictionary[wd] for wd in doc])

	pickle.dump(local_dictionary, open(corpus_name+'_'+'dictionary_local', 'wb'))
	pickle.dump(corpus_local_index, open(corpus_name+'_'+'corpus_local_index', 'wb'))

def preprocess_corpus_selected_from_global(corpus_flie_name):

	name = corpus_flie_name.split('_')
	corpus_name = name[0]+'_'+name[1]

	corpus = pickle.load(open(corpus_flie_name, 'rb'))
	corpus = corpus['pre_cutoff_data']

	gen_corpus_specific_dict(corpus, corpus_name)



################################################################################
### functions to convert the terms list between local index and global index ###
################################################################################

def local_to_global_term_index_list(local_term_index_list, local_dictionary):

	return [local_dictionary[term] for term in local_term_index_list]

def global_to_local_term_index_list(global_term_index_list, local_dictionary):

	revesed_local_dictionary = {local_dictionary[i]:i for i in local_dictionary}

	return [revesed_local_dictionary[term] for term in global_term_index_list]


######################################
### functions for final evaluation ###
######################################

def precision_at_k(k, ranked_list, corpus):

	cnt_k_cooccur_dict = count_all_terms_in_built_corpus(corpus)

	value = 0.0
	for i in range(k):
		if ranked_list[i] in cnt_k_cooccur_dict:
			value += 1

	return value/k

def spearsman_correlation(k, ranked_list, corpus, dict_stat, cutoff_year, normalized = False):

	term_index_list = ranked_list[:k]

	if normalized:
		sorting_index = rank_list_according_to_corpus_normalized(term_index_list, corpus, dict_stat, cutoff_year, after_cutoff=True)
	else:
		sorting_index = rank_list_according_to_corpus(term_index_list, corpus)

	ref_list = [i for i in range(k)]
	rho,pval = spearmanr(ref_list, sorting_index)

	return rho, pval

def precison_at_k_known_cooccur(k, rank_get, cnt_k_cooccur_dict):

	value = 0.0
	for i in range(k):
		if rank_get[i] in cnt_k_cooccur_dict:
			value += 1.0

	return value/k

def spearsman_correlation_at_k_two_ranks(k, rank_index_get, rank_index_ref):

	if k > len(rank_index_get):
		k = len(rank_index_get)

	rank_index_ref_dict = {rank_index_ref[i]:i for i in range(len(rank_index_ref))}
	ranks_of_get = [rank_index_ref_dict[rank_index_get[i]] for i in range(k)]
	rank_index_of_get = np.argsort(ranks_of_get)

	ref_list = [i for i in range(k)]
	rho, pval = spearmanr(ref_list, rank_index_of_get)

	return rho, pval
