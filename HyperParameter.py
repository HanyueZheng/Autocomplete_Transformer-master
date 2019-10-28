chunk_len=100   #每次训练用多少个字符
batch = 2        #每批几个输入输出
nbatches = 30       #每轮训练几批
epoch_number=1000 #要训练多少轮
epoches_of_loss_record=100 #每多少轮将损失输出到文本
epoches_of_model_save=1000   #每多少论存储一次模型
transformer_size=6  #encoder和decoder有多少层
predict_length=1   #预测结果的长度（字符数）
beam_search_number=3 #束搜索的大小


