import torch.nn as nn
from Sublayers import SublayerConnection, SublayerConnection4KG, SublayerConnection4AST
from Utils import clones
import torch


# 每层有两个子层:第一层是多头自注意机制，第二层是一个简单的、位置导向的、全连接的前馈网络。
class EncoderLayer(nn.Module):
	"编码器由以下的自注意力和前馈网络组成"

	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	#输入为数据及对应的mask，因为encoder中不需要掩盖任何输入token，所以mask一般为[1，1...1]
	#self_attn即multiheadattention，前三个输入为Query，Key，Value。其中Query可以理解为查询语句，Key-Value对可以理解为键值对，
	#最后一个参数mask，在decoder中会产生作用，因为在decoder中mask不全为1
	def forward(self, x, mask):
		"按照论文中的图1（左）的方式进行连接"
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)


# 在每个编码器层中的两个子层外，解码器还插入第三个子层，该子层在编码器堆栈的输出上执行多头关注。与编码器类似，使用残差连接解码器的每个子层，然后进行层归一化。
class DecoderLayer(nn.Module):
	"解码器由以下的自注意力、源注意力和前馈网络组成"

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"按照论文中的图1（右）的方式进行连接"
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)


class EncoderLayer4KG(nn.Module):
	def __init__(self, size, intermediate_size, self_attn, self_attn_ent, feed_forward, dropout):
		super(EncoderLayer4KG, self).__init__()
		self.attention = self_attn
		self.attention_ent = self_attn_ent
		self.sublayer = SublayerConnection4KG(size, intermediate_size, dropout)
		self.feed_forward = feed_forward
		self.size = size

	def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent=None, ent_mask=None):
		hidden_states, hidden_states_ent = hidden_states, hidden_states_ent
		# print("hidden_states.shape: ", hidden_states.shape)
		hidden_states = self.attention(hidden_states, hidden_states, hidden_states, attention_mask)
		# print(hidden_states_ent.shape)
		hidden_states_ent = self.attention_ent(hidden_states_ent, hidden_states_ent, hidden_states_ent, attention_mask_ent)
		# TODO
		# hidden_states_ent = hidden_states_ent  # * ent_mask
		return self.sublayer(hidden_states, hidden_states_ent)

class EncoderLayer4newAST(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer4newAST, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	# 输入为数据及对应的mask，因为encoder中不需要掩盖任何输入token，所以mask一般为[1，1...1]
	# self_attn即multiheadattention，前三个输入为Query，Key，Value。其中Query可以理解为查询语句，Key-Value对可以理解为键值对，
	# 最后一个参数mask，在decoder中会产生作用，因为在decoder中mask不全为1
	def forward(self, x, mask):
		"按照论文中的图1（左）的方式进行连接"
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

class EncoderLayer4AST(nn.Module):
	def __init__(self, size, intermediate_size, ast_size, self_attn, self_attn_ent, feed_forward, dropout, voc_size, embedding_dim , hidden_size, device):
		super(EncoderLayer4AST, self).__init__()
		self.attention = self_attn
		self.attention_ent = self_attn_ent
		self.num_lstm_layers = 2

		self.voc_size = ast_size
		self.hidden_size = hidden_size
		self.embedding_dim = embedding_dim
		self.device = device
		self.embedding = nn.Embedding(voc_size, embedding_dim)

		# forget gate
		# self.wf = nn.Linear(embedding_dim, hidden_size, bias=False)
		# self.uf = nn.Linear(hidden_size, hidden_size, bias=True)
        #
		# # input gate
		# self.wi = nn.Linear(embedding_dim, hidden_size, bias=False)
		# self.ui = nn.Linear(hidden_size, hidden_size, bias=True)
        #
		# # ouput gate
		# self.wo = nn.Linear(embedding_dim, hidden_size, bias=False)
		# self.uo = nn.Linear(hidden_size, hidden_size, bias=True)
        #
		# # for updating cell state vector
		# self.wc = nn.Linear(embedding_dim, hidden_size, bias=False)
		# self.uc = nn.Linear(hidden_size, hidden_size, bias=True)
        #
		# # gate's activation function
		# self.sigmoid = nn.Sigmoid()
        #
		# # activation function on the updated cell state
		# self.tanh = nn.Tanh()

		# distribution of the prediction

		self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers)
		self.out = nn.Linear(hidden_size, voc_size)
		self.softmax = nn.LogSoftmax(dim=0)

		self.sublayer = SublayerConnection4AST(ast_size, size, intermediate_size, dropout)
		self.feed_forward = feed_forward
		self.size = size

	def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, input, hidden, ent_mask=None):
		hidden_states, hidden_states_ent = hidden_states, hidden_states_ent
		hidden_states.size()
		hidden_states_ent.size()
		#input.size()
		hidden_states = self.attention(hidden_states, hidden_states, hidden_states, attention_mask)
		print(hidden_states.size())
		# print(hidden_states_ent.shape)
		hidden_states_ent = self.attention_ent(hidden_states_ent, hidden_states_ent, hidden_states_ent, attention_mask_ent)
		print(hidden_states_ent.size())

		input = input.long()
		input.size()
		#print(hidden.size())

		embed_input = self.embedding(input)
		#embed_input_b = self.embedding(input_b)

		#hidden_l = torch.chunk(hidden, 2, dim=-1)[0]
		#hidden_b = torch.chunk(hidden, 2, dim=-1)[1]

		# forget gate's activation vector
		# self.wf(embed_input).size()
		# self.uf(hidden).size()
		# f = self.sigmoid(self.wf(embed_input) + self.uf(hidden))
        #
        #
		# # input gate's activation vector
		# i = self.sigmoid(self.wi(embed_input) + self.ui(hidden))
        #
		# # output gate's activation vector
		# o = self.sigmoid(self.wo(embed_input) + self.uo(hidden))
		# tmp = self.tanh(self.wc(embed_input) + self.uc(hidden))
		# i.size()
		# tmp.size()
		# f.size()
		# cell_state.size()
		# updated_cell_state = torch.mul(cell_state, f) + torch.mul(i, tmp)
		# updated_hidden = torch.mul(self.tanh(updated_cell_state), o)

		#hidden.size()
		embed_input.size()
		#embed_input = embed_input.view(1, 1, -1)
		output, hidden = self.rnn(embed_input, hidden)
		output = self.softmax(self.out(output))

		#output = self.softmax(self.out(updated_hidden))
        #updated_hidden.size()
        #hidden_states_ent.size()


        #updated_cell_state.size()

		# print("output:")
		# print(output.size())
		# TODO
		# hidden_states_ent = hidden_states_ent  # * ent_mask
		x, ent, output = self.sublayer(hidden_states, hidden_states_ent, hidden)
		#return x, ent, output, updated_hidden, updated_cell_state
		return x, ent,output.view(-1),hidden
