# coding=utf-8
import torch
import sys
import seaborn
import json
from torch.autograd import Variable
from Batch import Batch, Batch_kg, Batch_ast
from Optim import NoamOpt, LabelSmoothing
from Model import make_model, make_model_kg, make_model_newast
from Train import run_epoch, greedy_decode, beam_search_decode, SimpleLossCompute, run_epoch_kg, beam_search_decode_kg, run_epoch_newast
from Dataloader import DataLoader_char, DataLoader_token, DataLoader_token_kg, Dataloader_token_newast
from HyperParameter import chunk_len, batch, nbatches, transformer_size, epoch_number, epoches_of_loss_record, \
    predict_length


seaborn.set_context(context="talk")


# cre文本匹配
# 输入数据处理
# 共有nbatches*batch*(chunklen-1)条数据
def data_gen_token_kg(dataloader, batch, nbatches, chunk_len, device):
    "为src-tgt复制任务生成随机数据"
    for i in range(nbatches):
        data_src = torch.empty(1, chunk_len - 1).long().to(device)
        data_ent = torch.empty(1, chunk_len - 1).long().to(device)
        data_tgt = torch.empty(1, 2).long().to(device)
        for k in range(batch):
            src_tgt_pair = dataloader.next_chunk()
            for j in range(0, len(src_tgt_pair)):
                data_src = torch.cat([data_src, src_tgt_pair[j][0].unsqueeze(0)])
                data_ent = torch.cat([data_ent, src_tgt_pair[j][1].unsqueeze(0)])
                data_tgt = torch.cat([data_tgt, src_tgt_pair[j][2].unsqueeze(0)])
            data_src = data_src[1:]
            data_ent = data_ent[1:]
            data_tgt = data_tgt[1:]
        src = Variable(data_src, requires_grad=False)
        ent = Variable(data_ent, requires_grad=False)
        tgt = Variable(data_tgt, requires_grad=False)
        yield Batch_kg(src, ent, tgt, -1)

def data_gen_token_newast(dataloader, batch, nbatches, chunk_len, device, astdim):
    "为src-tgt复制任务生成随机数据"
    for i in range(nbatches):
        data_src = torch.empty(1, chunk_len - 1).long().to(device)
        data_ent = torch.empty(1, chunk_len - 1).long().to(device)
        data_tgt = torch.empty(1, 2).long().to(device)
        data_ast = torch.empty(1, astdim).float().to(device)
        data_src.size()
        data_ast.size()
        for k in range(batch):
            src_tgt_pair = dataloader.next_chunk()
            for j in range(0, len(src_tgt_pair)):
                data_src = torch.cat([data_src, src_tgt_pair[j][0].unsqueeze(0)])
                src_tgt_pair[j][0].size()
                src_tgt_pair[j][3].size()
                data_ent = torch.cat([data_ent, src_tgt_pair[j][1].unsqueeze(0)])
                data_tgt = torch.cat([data_tgt, src_tgt_pair[j][2].unsqueeze(0)])
                data_ast = torch.cat([data_ast, src_tgt_pair[j][3].unsqueeze(0)])
            data_src = data_src[1:]
            data_ent = data_ent[1:]
            data_tgt = data_tgt[1:]
            data_ast = data_ast[1:]
        src = Variable(data_src, requires_grad=False)
        ent = Variable(data_ent, requires_grad=False)
        tgt = Variable(data_tgt, requires_grad=False)
        ast = Variable(data_ast, requires_grad=False)
        yield Batch_ast(src, ent, ast, tgt, -1)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # sys.argv.append('train')
    sys.argv.append('train')
    sys.argv.append('EN-ATP-V226.txt')
    sys.argv.append('token')
    sys.argv.append('Path.txt')
    #sys.argv.append('transformer200.model')
    #sys.argv.append('The ATP software is the core')

    if (len(sys.argv) < 2):
        print("usage: lstm [train file | inference (words vocabfile) ]")
        print("e.g. 1: lstm train cre-utf8.txt")
        print("e.g. 2: lstm inference cre-utf8.txt words")
        sys.exit(0)

    method = sys.argv[1]

    ents = []
    with open("kg_embed/entity2id.txt") as fin:
        fin.readline()
        for line in fin:
            name, id = line.strip().split("\t")
            ents.append(name)

    if (method == "train"):
        filename = sys.argv[2]
        is_char_level = sys.argv[3] == 'char'
        astfile = sys.argv[4]

        if is_char_level:
            # TODO
            dataloader = DataLoader_char(filename, chunk_len, device)
            V = dataloader.vocabularyLoader.n_chars  # vocabolary size

            # kg_embed
            # with open("kg_embed/embedding.vec.json", "r", encoding='utf-8') as f:
            #     lines = json.loads(f.read())
            #     vecs = list()
            #     # vecs.append([0] * 100)  # CLS
            #     for (i, line) in enumerate(lines):
            #         if line == "ent_embeddings":
            #             for vec in lines[line]:
            #                 vec = [float(x) for x in vec]
            #                 vecs.append(vec)
            # embed = torch.FloatTensor(vecs)
            # embed = torch.nn.Embedding.from_pretrained(embed)
            # print(embed)  # Embedding(464, 100)

            criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
            criterion.cuda()
            model = make_model(V, V, N=transformer_size)
            model.cuda()
            model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

            for epoch in range(epoch_number):
                if epoch % epoches_of_loss_record == 0:
                    f = open("procedure.txt", "a+")
                    f.write("step:%d \n" % epoch)
                    f.close()
                print("step: ", epoch)
                model.train()
                run_epoch("train", data_gen_char(dataloader, batch, nbatches), model,
                          SimpleLossCompute(model.generator, criterion, model_opt), nbatches, epoch)
                model.eval()
                run_epoch("test ", data_gen_char(dataloader, batch, 1), model,
                          SimpleLossCompute(model.generator, criterion, None), nbatches, epoch)

        else:

            #dataloader = DataLoader_token_kg(filename, ents, chunk_len, device)
            astfile = "Path.txt"
            dataloader = Dataloader_token_newast(filename, ents, astfile, chunk_len, device, astdim=99)
            V = dataloader.vocabularyLoader.n_tokens  # vocabolary size
            V_ast = dataloader.vocabularyLoader.n_token_newast

            criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
            #criterion.cuda()
            model = make_model_newast(V, V, V_ast, "kg_embed/embedding.vec.json", N=transformer_size)
            #model.cuda()
            model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

            for epoch in range(epoch_number):
                if epoch % epoches_of_loss_record == 0:
                    f = open("procedure.txt", "a+")
                    f.write("step:%d \n" % epoch)
                    f.close()
                print("step: ", epoch)
                model.train()
                run_epoch_newast("train", data_gen_token_newast(dataloader, batch, nbatches, chunk_len, device, astdim=99), model,
                          SimpleLossCompute(model.generator, criterion, model_opt), nbatches, epoch)
                model.eval()
                run_epoch_newast("test ", data_gen_token_newast(dataloader, batch, nbatches, chunk_len, device, astdim=99), model,
                          SimpleLossCompute(model.generator, criterion, None), nbatches, epoch)

    elif method == "inference":
        filename = sys.argv[2]
        is_char_level = sys.argv[3] == 'char'
        trained_model_name = sys.argv[4]
        words = sys.argv[5]

        if is_char_level:
            model = torch.load(trained_model_name).cuda()
            model.eval()
            dataloader = DataLoader_char(filename, chunk_len, device)
            src = Variable(dataloader.vocabularyLoader.char_tensor(words).unsqueeze(0))
            src_mask = Variable((src != 0).unsqueeze(-2))
            output_embed = greedy_decode(model, src, src_mask, max_len=predict_length)[0].cpu().numpy()
            result = ""
            for i in output_embed:
                result += dataloader.vocabularyLoader.index2char[i]
            print(result[1:])

        else:
            #model = torch.load(trained_model_name).cuda()
            model = torch.load(trained_model_name)
            model.eval()
            #dataloader = DataLoader_token_kg(filename, ents, chunk_len, device)
            dataloader = Dataloader_token_newast(filename, ents, astfile, chunk_len, device, astdim=99)
            V = dataloader.vocabularyLoader.n_tokens  # vocabolary size
            V_ast = dataloader.vocabularyLoader.n_token_newast

            word_list = words.replace('\n', ' ').replace('\t', ' ').split(' ')
            word_list = [i for i in word_list if (len(str(i))) != 0]
            src = Variable(dataloader.vocabularyLoader.token_tensor(word_list).unsqueeze(0))
            src_mask = Variable((src != 0).unsqueeze(-2))
            ent = Variable(torch.Tensor([24]*len(word_list)).long()).to(device)
            ents_list = []
            for i in range(len(dataloader.kg)):
                if words.find(" " + dataloader.kg[i] + " ") != -1:
                    ents_list.append(dataloader.kg[i])
            for i in range(len(ents_list)):
                key = ents_list[i].strip().split()
                if word_list.index(key[0]) >= 0:
                    ent[word_list.index(key[0])] = dataloader.kg.index(" ".join(key))
            ent = ent.unsqueeze(0)

            ent_mask = None

            output_embed_list = beam_search_decode_kg(model, src, src_mask, ent, ent_mask, max_len=predict_length)
            for j in range(len(output_embed_list)):
                output_embed = output_embed_list[j][1][0].cpu().numpy()
                result = []
                for i in output_embed:
                    result.append(dataloader.vocabularyLoader.index2token[i])
                result = result[1:]
                result = " ".join(result)
                print(result)