import torch
import torch.nn as nn
import math
from torch.autograd import Variable

# =============================================================================
#
# Embeddings and Softmax : 嵌入和SoftMax
#
# =============================================================================
# 根据给的词典大小，创建一个[vocab,d_model]大小的embedding，其中每一行代表一个词的embedding。
# dic={"hello":0,"world":1}，如果要看hello的Embedding，那么dic("hello")，即编号0被作为输入交给nn.Embedding，得到对应的Embedding。
# 这里只是初始的Embedding，要经过训练之后，才能得到hello真正的Embedding
class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x.long()) * math.sqrt(self.d_model)


# =============================================================================
#
# Positional Encoding : 位置编码
#
# =============================================================================
class PositionalEncoding(nn.Module):
	"实现位置编码函数"
	#PE(pos,2i) = sin(pos/10000^(2i/dmodel))
	#PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# 计算位置编码
		pe = torch.zeros(max_len, d_model)   #size是[max_len,d_model],全为0
		position = torch.arange(0., max_len).unsqueeze(1).float()  #size是[max_len,1],从0-max_len-1
		div_term = torch.exp((torch.arange(0., d_model, 2) * -math.log(10000) / d_model).float())  #size是[256]

		pe[:, 0::2] = torch.sin(position * div_term) #对pe中偶数列赋值
		pe[:, 1::2] = torch.cos(position * div_term) #对pe中奇数列赋值
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)


# # 位置编码将根据位置添加一个正弦波,波的频率和偏移对于每个维度都是不同的。
# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20, 0)
# y = pe.forward(Variable(torch.zeros(1, 10, 20)))
# plt.scatter(np.arange(10), y[0, :, 1].data.numpy())
# plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])