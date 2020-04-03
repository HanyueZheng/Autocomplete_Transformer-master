import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import Model
import matplotlib.pyplot as plt
import seaborn
from HyperParameter import epoches_of_loss_record, epoches_of_model_save, beam_search_number
import pdb
import inspect


# =============================================================================
#
# Training : 模型训练
#
# =============================================================================

# 接下来，我们创建一个通用的训练和评分功能来跟踪损失。我们传递一个通用的损失计算函数，该函数还处理参数更新。


def run_epoch(stage, data_iter, model, loss_compute, nbatches, epoch=0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss.detach().cpu().numpy()
        total_tokens += batch.ntokens.cpu().numpy()
        tokens += batch.ntokens.cpu().numpy()
        if i==nbatches-1:
            elapsed = time.time() - start
            print("%s  Loss: %f Tokens per Sec: %f" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
            if epoch%epoches_of_loss_record==0:
                f = open("procedure.txt", "a+")
                f.write("%s  Loss: %f Tokens per Sec: %f \n" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
                f.close()
            start = time.time()
            tokens = 0
    if epoch % epoches_of_model_save==0:
        torch.save(model, "transformer" + "{}".format(epoch) + ".model")


# 损失计算
class SimpleLossCompute:
    "一个简单的损失计算和训练函数"

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        print(x.contiguous().view(-1, x.size(-1)).type())
        print(y.contiguous().view(-1).float().type())
        try:
            print("normtype:")
            print(norm.float().type())
            print("ytype:")
            print(y.contiguous().view(-1).type())
            print(x.contiguous().view(-1, x.size(-1)).type())
            print("x.contiguous().view(-1, x.size(-1))")
            print(x.contiguous().view(-1, x.size(-1)))
            print("y.contiguous().view(-1).float()")
            print(y.contiguous().view(-1).float())
            import inspect

            inspect.getargspec(self.criterion)
            if self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1).float()) == None:
                print("None")
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1).float()) / norm.float()
        except Exception as e:
             pdb.set_trace()
             print(e)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()

# 贪婪解码
# 为了简单起见，此代码使用贪婪解码来预测。
def greedy_decode(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(src[0][-1].cpu().numpy().item()).type_as(src.data)
    for i in range(max_len):
        out = model.decode(memory, src_mask,Variable(ys),Variable(Model.Decoder.subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def beam_search_decode(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    ys = [0,torch.ones(1, 1).fill_(src[0][-1].cpu().numpy().item()).type_as(src.data)]
    reserved_options = choose_options(model, memory, src, src_mask, ys)
    for i in range(max_len-1):
        tmp_options=[]
        for j in range(len(reserved_options)):
            tmp_options+=choose_options(model,memory,src,src_mask,reserved_options[j])
        tmp_options=sorted(tmp_options,reverse=True)[:beam_search_number]
        reserved_options=tmp_options
    return reserved_options


def choose_options(model,memory,src,src_mask,ys):
    out = model.decode(memory, src_mask, Variable(ys[1]), Variable(Model.Decoder.subsequent_mask(ys[1].size(1)).type_as(src.data)))
    prob = model.generator(out[:, -1])
    dict = {}
    for j in range(prob.size()[-1]):
        dict[j] = prob[0][j].item()
    sort_dict = sorted(zip(dict.values(), dict.keys()), reverse=True)
    options = sort_dict[:beam_search_number]
    result=[]
    for i in range(beam_search_number):
        result.append([ys[0]+options[i][0],torch.cat([ys[1], torch.ones(1, 1).type_as(src.data).fill_(options[i][1])], dim=1)])
    return result


def run_epoch_kg(stage, data_iter, model, loss_compute, nbatches, epoch=0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.ent, batch.trg, batch.src_mask, batch.ent_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss.detach().cpu().numpy()
        total_tokens += batch.ntokens.cpu().numpy()
        tokens += batch.ntokens.cpu().numpy()
        if i == nbatches-1:
            elapsed = time.time() - start
            print("%s  Loss: %f Tokens per Sec: %f" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
            if epoch%epoches_of_loss_record==0:
                f = open("procedure.txt", "a+")
                f.write("%s  Loss: %f Tokens per Sec: %f \n" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
                f.close()
            start = time.time()
            tokens = 0
    if epoch%epoches_of_model_save==0:
        torch.save(model, "transformer" + "{}".format(epoch) + ".model")

def run_epoch_newast(stage, data_iter, model, loss_compute, nbatches, epoch=0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.ent, batch.trg, batch.src_mask, batch.ent_mask, batch.trg_mask, batch.ast, batch.ast_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss.detach().cpu().numpy()
        total_tokens += batch.ntokens.cpu().numpy()
        tokens += batch.ntokens.cpu().numpy()
        if i == nbatches-1:
            elapsed = time.time() - start
            print("%s  Loss: %f Tokens per Sec: %f" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
            if epoch%epoches_of_loss_record==0:
                f = open("procedure.txt", "a+")
                f.write("%s  Loss: %f Tokens per Sec: %f \n" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
                f.close()
            start = time.time()
            tokens = 0
    if epoch%epoches_of_model_save==0:
        torch.save(model, "transformer" + "{}".format(epoch) + ".model")

def run_epoch_ast(stage, data_iter, model, loss_compute, nbatches, device, ast_token_num, embedding_dim=512, hidden_size=512, epoch=0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    print("run_epoch_ast")
    lstm = Model.LSTM(ast_token_num ,embedding_dim, hidden_size)
    hidden = lstm.init_hidden(nbatches, device)
    #cell_state = lstm.init_cell_state().to(device)
    #cell_state_b = lstm.init_cell_state().to(device)

    for i, batch in enumerate(data_iter):
        #print(batch.src.size())
        #print(batch.ent.size())
        # print(batch.trg.size())
        # print(batch.ast.size())
        out = model.forward(batch.src, batch.ent, batch.trg, batch.src_mask, batch.ent_mask, batch.trg_mask, batch.ast, hidden)
        print("forward down")
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss.detach().cpu().numpy()
        total_tokens += batch.ntokens.cpu().numpy()
        tokens += batch.ntokens.cpu().numpy()
        if i == nbatches-1:
            elapsed = time.time() - start
            print("%s  Loss: %f Tokens per Sec: %f" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
            if epoch%epoches_of_loss_record==0:
                f = open("procedure.txt", "a+")
                f.write("%s  Loss: %f Tokens per Sec: %f \n" % (stage, loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy(), tokens / elapsed))
                f.close()
            start = time.time()
            tokens = 0
    if epoch%epoches_of_model_save==0:
        torch.save(model, "transformer" + "{}".format(epoch) + ".model")


def beam_search_decode_kg(model, src, src_mask, ent, ent_mask, max_len):
    memory = model.encode(src, src_mask, ent, ent_mask)
    ys = [0, torch.ones(1, 1).fill_(src[0][-1].cpu().numpy().item()).type_as(src.data)]
    reserved_options = choose_options(model, memory, src, src_mask, ys)
    for i in range(max_len-1):
        tmp_options = []
        for j in range(len(reserved_options)):
            tmp_options += choose_options(model, memory, src, src_mask, reserved_options[j])
        tmp_options = sorted(tmp_options,reverse=True)[:beam_search_number]
        reserved_options = tmp_options
    return reserved_options

def beam_search_decode_ast(model, src, src_mask, ent, ent_mask, ast, hidden, cell_state, max_len):
    src.size()
    ent.size()
    ast.size()
    memory = model.encode(src, src_mask, ent, ent_mask, ast, hidden, cell_state)
    ys = [0, torch.ones(1, 1).fill_(src[0][-1].cpu().numpy().item()).type_as(src.data)]
    reserved_options = choose_options(model, memory, src, src_mask, ys)
    for i in range(max_len - 1):
        tmp_options = []
        for j in range(len(reserved_options)):
            tmp_options += choose_options(model, memory, src, src_mask, reserved_options[j])
        tmp_options = sorted(tmp_options, reverse=True)[:beam_search_number]
        reserved_options = tmp_options
    return reserved_options
