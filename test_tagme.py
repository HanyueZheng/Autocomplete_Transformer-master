# import tagme
# tagme.GCUBE_TOKEN = "49cef025-1fec-4bf7-99a1-2d0957843472-843339462"
#
# text = "ATP read CCdataPlugInfo from the CC data plug whose structure as shown in Table " \
#        " 4  4  .  The safety-related information used by ATP are coded by VCP and the other " \
#        "information such IP address sent to DLU or CCNV are not need to encode . "
# text_ann = tagme.annotate(text)
# print(text_ann)
#
#
# ent_map = {}
# ents = []
# with open("kg_embed/entity2id.txt") as fin:
#     fin.readline()
#     for line in fin:
#         name, id = line.strip().split("\t")
#         hash(name)
#         ent_map[name] = id
#         ents.append(name)

# kg_embed
import json
import torch
with open("kg_embed/embedding.vec.json", "r", encoding='utf-8') as f:
    lines = json.loads(f.read())
    vecs = list()
    # vecs.append([0] * 100)  # CLS
    for (i, line) in enumerate(lines):
        if line == "ent_embeddings":
            for vec in lines[line]:
                vec = [float(x) for x in vec]
                vecs.append(vec)
embed = torch.FloatTensor(vecs)
embed = torch.nn.Embedding.from_pretrained(embed)

text = torch.Tensor([23,34]).long()
text = embed(text)
print(text)

# def get_ents(ann):
#     ents = []
#     # Keep annotations with a score higher than 0.3
#     for a in ann.get_annotations(-1):
#         print(a)
#         if a.entity_title in ent_map:
#             # print(a.entity_title)
#             ents.append([ent_map[a.entity_title], a.begin, a.end, a.score])
#         elif a.mention in ent_map:
#             # print(a.mention)
#             ents.append([ent_map[a.mention], a.begin, a.end, a.score])
#         else:
#             continue
#     return ents
#
# ents = get_ents(text_ann)
# print(ents)

# for i in range(len(ents)):
#     if text.find(ents[i]) > 0:
#         print(ents[i])
#         print(text.find(ents[i]))
