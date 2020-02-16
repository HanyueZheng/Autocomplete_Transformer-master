import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Utils import clones


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# 在两个子层中的每一个子层使用一个残差连接，然后进行层归一化
class LayerNorm(nn.Module):
	"构建层归一化模块"

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 每一个子层的输出为 LayerNorm(x+Sublayer(x))，其中 Sublayer(x) 是子层自己实现的函数，在子层输入和归一化之前完成每一个子层输出的dropout。
# 为了实现残差连接，模型中的所有子层以及嵌入层的输出维度都是dmodel=512。
class SublayerConnection(nn.Module):
	"""
	层归一化之后的残差连接。
	注意：为了简化代码，归一化是第一个，而不是最后一个。
	"""

	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"将残差连接应用于相同大小的任何子层。"
		return x + self.dropout(sublayer(self.norm(x)))


# =============================================================================
#
# Attention 注意力
#
# =============================================================================
# 公式：Attention(Q,K,V)=softmax(Q K^T /√dk ) V
# 按照论文中给出的计算过程写即可
def attention(query, key, value, mask=None, dropout=None):
	"计算'可缩放点乘注意力'"
	d_k = query.size(-1)
	# 此处query和key的大小为[nbatches,head_size,src_size,d_moel/head_size]，经过torch.matmul后大小为[nbatches,head_size,src_size,src_size]
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


# 多头注意力模型,输入张量维数为512,计划分为8个头来进行处理,即每个张量分成8个64维的张量来处理
class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"设置模型大小和注意力头部数量"
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# 假设 d_v 等于 d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)  # 对应 Q,K,V 3次线性变换 + 最终的1次线性变换
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		"实现论文中的第2张图"
		if mask is not None:
			# 同样的屏蔽适用于所有h型头
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# 1)批量执行所有线性变换 d_model => h x d_k
		query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
		                     for l, x in zip(self.linears, (query, key, value))]

		# 2）将注意力集中在批量的所有投射向量上
		x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

		# 3)使用view方法做Concat然后做最终的线性变换。
		# 处理前x的大小为[nbatches,self_h,src_size,self.d_k],处理后的大小为[nbatches,src_size,d_model]
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

		return self.linears[-1](x)


# =============================================================================
#
# Position-wise Feed-Forward Networks 位置前馈网络
#
# =============================================================================
# 计算公式：FFN(x)=max(0,xW1+b1)W2+b2  论文中 输入、输出的维度dmodel=512, 内部隐藏层的维度 dff=2048.
class PositionwiseFeedForward(nn.Module):
	"实现FFN方程"

	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)


	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
	"定义标准linear + softmax 步骤"

	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)


class IntermediateLayer(nn.Module):
	def __init__(self, size, intermediate_size):
		super(IntermediateLayer, self).__init__()
		self.dense = nn.Linear(size, intermediate_size)
		self.dense_ent = nn.Linear(512, intermediate_size)
		self.aa = nn.Linear(99, intermediate_size)

		# Information Fusion
		self.intermediate_act_fn = gelu

	def forward(self, hidden_states, hidden_states_ent):
		# print("hidden_states:")
		print(hidden_states.size())
		# print("hidden_states_ent size:")
		print(hidden_states_ent.size())
		# print("hidden_states_ast:")
		#print(hidden_states_ast.size())

		hidden_states_ = self.dense(hidden_states)
		hidden_states_.size()
		hidden_states_ent_ = self.dense_ent(hidden_states_ent)
		hidden_states_ent_.size()

		#hidden_states_ast.size()
		# print("hidden_states size:")
		# print(hidden_states.size())
		# print("hidden_states_ent size:")
		# print(hidden_states_ent.size())
		# print("hidden_states_ast:")
		# print(hidden_states_ast.size())


		hidden_states = self.intermediate_act_fn(hidden_states_ + hidden_states_ent_)

		return hidden_states#, hidden_states_ent


class SublayerConnection4KG(nn.Module):
	def __init__(self, size, intermediate_size, dropout):
		super(SublayerConnection4KG, self).__init__()
		self.intermediate = IntermediateLayer(size, intermediate_size)
		self.dense = nn.Linear(size, intermediate_size)
		self.dense_ent = nn.Linear(512, intermediate_size)
		self.norm = LayerNorm(size)
		self.norm_ent = LayerNorm(512)
		self.dropout = nn.Dropout(dropout)

	def forward(self, attention_output, attention_output_ent):
		intermediate_output = self.intermediate(attention_output, attention_output_ent)
		hidden_states = self.dense(intermediate_output)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.norm(hidden_states + attention_output)

		hidden_states_ent = self.dense_ent(intermediate_output)
		hidden_states_ent = self.dropout(hidden_states_ent)
		hidden_states_ent = self.norm_ent(hidden_states_ent + attention_output_ent)

		return hidden_states, hidden_states_ent

class SublayerConnection4AST(nn.Module):
	def __init__(self, ast_size, size, intermediate_size, dropout):
		super(SublayerConnection4AST, self).__init__()
		self.intermediate = IntermediateLayer(ast_size, size, intermediate_size)
		self.dense = nn.Linear(size, intermediate_size)
		self.dense_ent = nn.Linear(512, intermediate_size)
		self.dense_ast = nn.Linear(512, intermediate_size)
		self.dense_outputast = nn.Linear(931, intermediate_size)

		self.norm = LayerNorm(size)
		self.norm_ent = LayerNorm(512)
		self.norm_ast = LayerNorm(512)
		self.dropout = nn.Dropout(dropout)

	def forward(self, attention_output, attention_output_ent, output_ast):
		#output_ast.size()
		intermediate_output = self.intermediate(attention_output, attention_output_ent, output_ast)
		intermediate_output.size()
		hidden_states = self.dense(intermediate_output)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.norm(hidden_states + attention_output)

		hidden_states_ent = self.dense_ent(intermediate_output)
		hidden_states_ent = self.dropout(hidden_states_ent)
		hidden_states_ent = self.norm_ent(hidden_states_ent + attention_output_ent)

		hidden_states_ast = self.dense_ast(intermediate_output)
		hidden_states_ast = self.dropout(hidden_states_ast)
		hidden_states_ast.size()

		output_ast = output_ast.permute(0,2,1)
		output_ast = self.dense_outputast(output_ast)
		hidden_states_ast = self.norm_ast(hidden_states_ast + output_ast)
		output_ast.size()

		return hidden_states, hidden_states_ent, hidden_states_ast
