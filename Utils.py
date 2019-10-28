from torch import nn
import copy

# 论文中的编码器由N=6个相同层的堆栈组成
def clones(module, N):
	"生成N个相同的层"
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])