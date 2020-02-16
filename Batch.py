from torch.autograd import Variable
from Model import Decoder
import torchsnooper


# Batches and Masking
class Batch:
    "此对象用于在训练时进行已屏蔽的批数据处理"

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个mask来隐藏填充和将来的单词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# 我们将使用torch 文本进行批处理。在TorchText函数中创建批次，确保填充最大批次大小不超过阈值（如果我们有8个GPU，则为25000）。
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "持续扩大批处理并计算标识+填充的总数"
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.src) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class Batch_kg:
    "此对象用于在训练时进行已屏蔽的批数据处理"

    def __init__(self, src, ent, trg=None, pad=0):
        self.src = src
        self.ent = ent
        self.trg = trg
        self.src_mask = (src != pad).unsqueeze(-2)
        self.ent_mask = None
        if self.trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个mask来隐藏填充和将来的单词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            Decoder.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class Batch_ast:
    "此对象用于在训练时进行已屏蔽的批数据处理"

    def __init__(self, src, ent, ast, trg=None, pad=0):
        self.src = src
        self.ent = ent
        self.trg = trg
        self.ast = ast
        self.src_mask = (src != pad).unsqueeze(-2)
        self.ent_mask = None
        self.ast_mask = (src != pad).unsqueeze(-2)
        if self.trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个mask来隐藏填充和将来的单词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            Decoder.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask