import torch.nn as nn
import numpy as np
import torch
import copy
import json
from Sublayers import MultiHeadedAttention, PositionwiseFeedForward, Generator, LayerNorm
from Layers import EncoderLayer, DecoderLayer, clones
from Embed import Embeddings, PositionalEncoding
from Layers import EncoderLayer4KG, EncoderLayer4AST
from torch.autograd import Variable

# =============================================================================
#
# Full Model : 整体模型
#
# =============================================================================
# 定义一个函数，它接受超参数并生成完整的模型。
# Transformer由encoder和decoder组成。其中用到的sublayer有MultiHeadedAttention，PositionwiseFeedForward，这两个是在encoder和decoder中的，
# 然后PositionalEncoding和Embeddings是用在输入之后，encoder及decoder层之间的，Generator是用在decoder之后的
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "从超参数构造模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 从代码来看，使用 Glorot / fan_avg初始化参数很重要。
    # 对参数进行均匀分布初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class EncoderDecoder(nn.Module):
    """
    标准编码器-解码器结构，本案例及其他各模型的基础。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "处理屏蔽的源序列与目标序列"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# =============================================================================
#
# Encoder 编码器
#
# =============================================================================
class Encoder(nn.Module):
    "核心编码器是N层堆叠"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "依次将输入的数据（及屏蔽数据）通过每个层"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
#
# Decoder 解码器
#
# =============================================================================
# 解码器也由一个N=6个相同层的堆栈组成。
class Decoder(nn.Module):
    "带屏蔽的通用N层解码器"

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

    def subsequent_mask(size):
        "屏蔽后续位置"
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0


# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])


def make_model_kg(src_vocab, tgt_vocab, kg_embed, N=6, d_model=512, d_ff=2048, d_intermediate=512, h=8, dropout=0.1):
    "从超参数构造模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    attn_ent = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    with open(kg_embed, "r", encoding='utf-8') as f:
        lines = json.loads(f.read())
        vecs = list()
        # vecs.append([0] * 100)  # CLS
        for (i, line) in enumerate(lines):
            if line == "ent_embeddings":
                for vec in lines[line]:
                    vec = [float(x) for x in vec]
                    vecs.append(vec)
    embed = torch.FloatTensor(vecs)

    model = EncoderDecoder4KG(
        Encoder4KG(EncoderLayer4KG(d_model, d_intermediate, c(attn), c(attn_ent), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        torch.nn.Embedding.from_pretrained(embed),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 从代码来看，使用 Glorot / fan_avg初始化参数很重要。
    # 对参数进行均匀分布初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class EncoderDecoder4KG(nn.Module):
    """
    标准编码器-解码器结构，本案例及其他各模型的基础。
    """

    def __init__(self, encoder, decoder, src_embed, ent_embed, tgt_embed, generator):
        super(EncoderDecoder4KG, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.ent_embed = ent_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, ent, tgt, src_mask, ent_mask, tgt_mask):
        "处理屏蔽的源序列与目标序列"
        return self.decode(self.encode(src, src_mask, ent, ent_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask, ent, ent_mask):
        return self.encoder(self.src_embed(src), src_mask, self.ent_embed(ent), ent_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder4KG(nn.Module):
    "核心编码器是N层堆叠"

    def __init__(self, layer, N):
        super(Encoder4KG, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, ent, ent_mask):
        "依次将输入的数据（及屏蔽数据）通过每个层"
        for layer in self.layers:
            x, ent = layer(x, mask, ent, ent_mask)
        return self.norm(x)


def make_model_ast(src_vocab, tgt_vocab, voc_size, device, kg_embed, ast_embed, N=6, d_model=512, d_ff=2048,
                   d_intermediate=512, h=8, dropout=0.1, embedding_dim=99, hidden_size=99):
    # "????????
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    attn_ent = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    with open(kg_embed, "r", encoding='utf-8') as f:
        lines = json.loads(f.read())
        vecs = list()
        # vecs.append([0] * 100)  # CLS
        for (i, line) in enumerate(lines):
            if line == "ent_embeddings":
                for vec in lines[line]:
                    vec = [float(x) for x in vec]
                    vecs.append(vec)
    embed = torch.FloatTensor(vecs)

    ast_size = voc_size
    path_listtensor = list()
    lstm = LSTM(voc_size, embedding_dim, hidden_size)
    # bilstm = BiLSTM(voc_size, embedding_dim, hidden_size, device)

    # for k in ast_embed.keys():
    #     pathlist = ast_embed[k]
    #     path_tensor = torch.zeros(voc_size).long()
    #     path_tensor = Variable(path_tensor).to(device)
    #     for p in pathlist:
    #         outpath = torch.zeros(voc_size).long()
    #         outpath = Variable(outpath).to(device)
    #         p.unsqueeze(0)
    #         hidden = lstm.init_hidden().to(device)
    #         cell_state = lstm.init_cell_state().to(device)
    #         #cell_state_b = lstm.init_cell_state().to(device)
    #         i = 0
    #         while i < (list(p.size())[0]):
    #             # output, hiddenout, cell_state, cell_state_b = bilstm.forward(p[i], p[list(p.size())[0] - 1 - i], hidden, cell_state, cell_state_b)
    #             output, hidden, cell_state = lstm(p[i], hidden, cell_state)
    #             i += 1
    #             outpath += output
    #         path_tensor += outpath
    #
    #     path_float = list()
    #     for x in path_tensor:
    #         path_float.append(float(x))
    #     path_listtensor.append(path_float)
    # ast_embed = torch.FloatTensor(path_listtensor)

    model = EncoderDecoder4AST(
        Encoder4AST(EncoderLayer4AST(d_model, d_intermediate, ast_size, c(attn), c(attn_ent), c(ff), dropout, voc_size,
                                     embedding_dim, hidden_size, device), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        torch.nn.Embedding.from_pretrained(embed),
        #nn.Sequential(Embeddings(d_model, voc_size)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # ?????,?? Glorot / fan_avg?????????
    # ????????????
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Encoder4AST(nn.Module):
    "核心编码器是N层堆叠"

    def __init__(self, layer, N):
        super(Encoder4AST, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, ent, ent_mask, input, hidden, cell_state):
        "依次将输入的数据（及屏蔽数据）通过每个层"
        # print("x_size:")
        # print(x.size())
        # print("ent_size:")
        # print(ent.size())
        # print(input.size())
        for layer in self.layers:
            x, ent, output, hidden, cell_state = layer(x, mask, ent, ent_mask, input, hidden,
                                                                     cell_state)
        return self.norm(x)


class EncoderDecoder4AST(nn.Module):
    """
    标准编码器-解码器结构，本案例及其他各模型的基础。
    """

    def __init__(self, encoder, decoder, src_embed, ent_embed, tgt_embed, generator):
        super(EncoderDecoder4AST, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.ent_embed = ent_embed
        #self.ast_embed = ast_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, ent, tgt, src_mask, ent_mask, tgt_mask, input, hidden, cell_state):
        "处理屏蔽的源序列与目标序列"

        return self.decode(self.encode(src, src_mask, ent, ent_mask, input, hidden, cell_state),
                           src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask, ent, ent_mask, input, hidden, cell_state):
        return self.encoder(self.src_embed(src), src_mask, self.ent_embed(ent), ent_mask, input, hidden,
                            cell_state)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# class BiLSTM(nn.Module):
#     # 用于编码ast
#     def __init__(self, voc_size, embedding_dim, hidden_size, device):
#         super(BiLSTM, self).__init__()
#         self.voc_size = voc_size
#         self.hidden_size = hidden_size
#         self.embedding_dim = embedding_dim
#         self.device = device
#
#         self.embedding = nn.Embedding(voc_size, embedding_dim)
#
#         # forget gate
#         self.wf = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.uf = nn.Linear(hidden_size, hidden_size, bias=True)
#
#         # forget gate back
#         self.wfb = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.ufb = nn.Linear(hidden_size, hidden_size, bias=False)
#
#         # input gate
#         self.wi = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.ui = nn.Linear(hidden_size, hidden_size, bias=True)
#
#         # input gate back
#         self.wib = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.uib = nn.Linear(hidden_size, hidden_size, bias=True)
#
#         # ouput gate
#         self.wo = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.uo = nn.Linear(hidden_size, hidden_size, bias=True)
#
#         # output gate back
#         self.wob = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.uob = nn.Linear(hidden_size, hidden_size, bias=True)
#
#         # for updating cell state vector
#         self.wc = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.uc = nn.Linear(hidden_size, hidden_size, bias=True)
#
#         # update cell back
#         self.wcb = nn.Linear(embedding_dim, hidden_size, bias=False)
#         self.ucb = nn.Linear(hidden_size, hidden_size, bias=True)
#
#         # gate's activation function
#         self.sigmoid = nn.Sigmoid()
#
#         # activation function on the updated cell state
#         self.tanh = nn.Tanh()
#
#         # distribution of the prediction
#         self.out = nn.Linear(hidden_size * 2, voc_size)
#         self.softmax = nn.LogSoftmax(dim=0)
#
#     def forward(self, input, input_b, hidden, cell_state, cell_state_b):
#         embed_input = self.embedding(input)
#         embed_input_b = self.embedding(input_b)
#
#         hidden_l = torch.chunk(hidden, 2, dim=-1)[0]
#         hidden_b = torch.chunk(hidden, 2, dim=-1)[1]
#         # @ hidden_b = torch.chunk(hidden, 2, dim = 1)[1]
#         # print(hidden_b)
#         # forget gate's activation vector
#         f = self.sigmoid(self.wf(embed_input) + self.uf(hidden_l))
#         f_b = self.sigmoid(self.wf(embed_input_b) + self.uf(hidden_b))
#
#         # input gate's activation vector
#         i = self.sigmoid(self.wi(embed_input) + self.ui(hidden_l))
#         c_hat = self.tanh(self.wc(embed_input) + self.uc(hidden_l))
#
#         i_b = self.sigmoid(self.wib(embed_input_b) + self.uib(hidden_b))
#         c_hat_b = self.tanh(self.wc(embed_input_b) + self.uc(hidden_b))
#
#         updated_cell_state = torch.mul(cell_state, f) + torch.mul(i, c_hat)
#         updated_cell_state_b = torch.mul(cell_state_b, f_b) + torch.mul(i_b, c_hat_b)
#
#         # output gate's activation vector
#         o = self.sigmoid(self.wo(embed_input) + self.uo(hidden_l))
#         updated_hidden = torch.mul(self.tanh(updated_cell_state), o)
#
#         o_b = self.sigmoid(self.wob(embed_input_b) + self.uob(hidden_b))
#         updated_hidden_b = torch.mul(self.tanh(updated_cell_state_b), o_b)
#
#         hiddenout = torch.cat([updated_hidden, updated_hidden_b], dim=-1)
#         output = self.softmax(self.out(hiddenout))
#         return output, hiddenout, updated_cell_state, updated_cell_state_b
#
#     def init_hidden(self):
#         return Variable(torch.zeros(self.hidden_size * 2))
#
#     def init_cell_state(self):
#         return Variable(torch.zeros(self.hidden_size))


class LSTM(nn.Module):
    def __init__(self, voc_size, embedding_dim, hidden_size):
        super(LSTM, self).__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(voc_size, embedding_dim)
        # forget gate
        self.wf = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.uf = nn.Linear(hidden_size, hidden_size, bias=True)
        # input gate
        self.wi = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.ui = nn.Linear(hidden_size, hidden_size, bias=True)
        # ouput gate
        self.wo = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.uo = nn.Linear(hidden_size, hidden_size, bias=True)
        # for updating cell state vector
        self.wc = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.uc = nn.Linear(hidden_size, hidden_size, bias=True)
        # gate's activation function
        self.sigmoid = nn.Sigmoid()
        # activation function on the updated cell state
        self.tanh = nn.Tanh()
        # distribution of the prediction
        self.out = nn.Linear(hidden_size, voc_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden, cell_state):
        embed_input = self.embedding(input)
        # forget gate's activation vector
        f = self.sigmoid(self.wf(embed_input) + self.uf(hidden))
        # input gate's activation vector
        i = self.sigmoid(self.wi(embed_input) + self.ui(hidden))
        # output gate's activation vector
        o = self.sigmoid(self.wo(embed_input) + self.uo(hidden))
        tmp = self.tanh(self.wc(embed_input) + self.uc(hidden))
        updated_cell_state = torch.mul(cell_state, f) + torch.mul(i, tmp)
        updated_hidden = torch.mul(self.tanh(updated_cell_state), o)
        output = self.softmax(self.out(updated_hidden))
        return output, updated_hidden, updated_cell_state

    def init_hidden(self):
        return Variable(torch.zeros(self.hidden_size))

    def init_cell_state(self):
        return Variable(torch.zeros(self.hidden_size))
