import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn.pytorch.glob import SetTransformerEncoder
from dgl.nn.pytorch.glob import SetTransformerDecoder

torch.backends.cudnn.benchmark=False
CUDA_LAUNCH_BLOCKING="1"

def set2dgraph(set_as_list):
	g = dgl.DGLGraph()
	g.add_nodes(len(set_as_list), {'i': torch.tensor(set_as_list)})
	return g

def articles2batch(articles_list, device):

	articles_dgl_graph = []

	for article in articles_list:
		g = set2dgraph(article)
		articles_dgl_graph.append(g)

	batch_g = dgl.batch(articles_dgl_graph)

	return batch_g.to(device)

def tile(a, n_tile_list):
	output = []
	for i in range(len(a)):
		ts = a[i]
		rpt = n_tile_list['_U'][i]
		output.append(ts.repeat((rpt,1)))
	return torch.cat(tuple(output),dim=0)

class MessagePassingCell(nn.Module):
	"""docstring for ClassName"""
	def __init__(self, hidden_size):
		super(MessagePassingCell, self).__init__()

		self.hidden_size = hidden_size

		self.pool = SumPooling()
		# self.pool = AvgPooling()

		# self.pool = GlobalAttentionPooling(
		#	gate_nn=
		#		nn.Sequential(
		#			nn.Linear(hidden_size, 1)
		#		),
		#	feat_nn=
		#		nn.Sequential(
		#			nn.Linear(hidden_size, hidden_size),
		#			nn.ReLU(),
		#			nn.Linear(hidden_size, hidden_size),
		#			nn.ReLU()
		#		),
		#
		#)

		self.setupdate = nn.Sequential(
			nn.Linear(hidden_size*2, hidden_size),
			nn.ReLU(),
			# nn.Linear(hidden_size, hidden_size),
			# nn.ReLU(),
			# nn.Linear(hidden_size, hidden_size),
			# nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU()
		)

		self.nodeupdate = nn.Sequential(
			nn.Linear(hidden_size*2, hidden_size),
			nn.ReLU(),
			# nn.Linear(hidden_size, hidden_size),
			# nn.ReLU(),
			# nn.Linear(hidden_size, hidden_size),
			# nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU()
		)

		#self.to(torch.device('cuda:0'))

	def forward(self, graph, updated_set, updated_node):
		
		nodes_agg = self.pool(graph, updated_node)
		set_update_input = torch.cat((nodes_agg, updated_set), dim=1)
		updated_set = self.setupdate(set_update_input)

		node_update_input = torch.cat((updated_node, tile(updated_set, graph._batch_num_nodes)), dim=1)
		updated_node = self.nodeupdate(node_update_input)

		graph.ndata['h'] = updated_node

		return graph, updated_set, updated_node
		

class HGRanking(nn.Module):
	def __init__(self, vocab_size, hidden_size, mp_round, dropout, device):
		super(HGRanking, self).__init__()

		self.hidden_size = hidden_size

		self.mp_round = mp_round

		self.embedding = nn.Embedding(vocab_size, hidden_size)

		self.mp = MessagePassingCell(hidden_size)

		self.dropout = nn.Dropout(dropout)

		self.readout = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh()
		)

		self.device = device

		self.to(device) #torch.device('cuda:0')

	def forward(self, articles_list):

		graph = articles2batch(articles_list, self.device)

		batch_size = graph.batch_size

		updated_set = torch.cat(batch_size*[torch.ones(1,self.hidden_size)])
		updated_node = self.dropout(self.embedding(graph.ndata['i']))

		updated_set = updated_set.to(self.device)
		updated_node = updated_node.to(self.device)

		for i in range(self.mp_round):
			graph, updated_set, updated_node = self.mp(graph, updated_set, updated_node) #self.dropout(updated_set), self.dropout(updated_node))
		
		output = self.readout(updated_set)
		output = torch.norm(output, dim=1)

		return output
