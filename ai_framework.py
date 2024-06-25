# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/4/25


"""
    主要的框架为：
        1. Load Data: dataset, dataloader
        2. 训练3件套： net, loss function, optimizer
        3.
"""


""" 
    重要的tensor ops
        squeeze(ndim): ndim被去掉的维度
        unsqueeze(ndim)： 增加新维度
        cat： concat
        x.reshape/x.view
        x.to('cuda')
"""


"""
    Loss Functions
    nn.MSELoss()
    nn.CrossEntropyLoss()
    
"""


"""
    搭建网络方法
    1. nn.Sequential()
    2. forward 传递
"""


"""
    optimizer
    torch.optim.SGD(model.parameters(), lr, momentum=0)
"""

"""
    标准模板：
    # load data
        dataset = MyDataset(file)   # read data via MyDataset
        dv_set = DataLoader(dataset, 16,shuffle=True)    #put dataset into Dataloader, 其实类似一个iter对象
    # set training module
        model = MyModel().to(device)      # contruct model and move to device(cpu/cuda)
        criterion =nn.MSELOSS()       # set loss function
        optimizer =torch.optim.SGD(model.parameters()，0.1)    # set optimizer
    # train process
        for epoch in range(n_epochs):    # iterate n epochs
            model.train()      # set model to train mode
            for x,y in tr set:     # iterate through the dataloader
                optimizer.zero grad()    # set gradient to zero
                x，y=x.to(device),y.to(device)    # move data to device (cpu/cuda)
                pred = model(x)    # forward pass (compute output)
                loss = criterion(pred, y)     # compute loss
                loss.backward()    # compute gradient(backpropagation)
                optimizer.step()    # update model with optimizer
    # validation process（和train在一个loop里）
            model.eval()       # set model to evaluation mode, BN and dropout turned off
            total loss =0      # iterate through the dataloader
            for x,y in dv_set:   
                x，y=x.to(device),y.to(device)    # move data to device(cpu/cuda)
                with torch.no_grad():             # disable gradient calculation
                    pred = model(x)                  # forward pass(compute output)
                    loss = criterion(pred, y)        # compute loss
                total loss +=loss.cpu().item()* len(x)      # accumulate loss
                avg loss = total loss /len(dv set.dataset)   # compute averaged loss
    # ps: train和valid的部分很多时候可以放train函数里面
    比如：train(net, train_iter, test_iter, loss, num_epochs, trainer)  # d2l
    # test process(和train和valid独立的loop)
        model.eval()                        # set model to evaluation mode
        preds =[]                           # iterate through the dataloader
        for xin tt_set:
            x= x.to(device)                 # move data to device(cpu/cuda)
            with torch.no_grad():           # disable gradient calculation
                pred = model(x)             # forward pass(compute output)
                preds.append(pred.cpu())    # collect prediction
    # save and load model
        torch.save(model.state_dict(), path)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt)
"""

"""
    有用的工具
    torchaudio：speech/audio processing
    torchtext：natural language processing
    torchvision: computer vision
    skorch: scikit-learn + pyTorch
    Huggingface Transformers (transformer models: BERT, GPT, ...)
    Fairseg (sequence modeling for NLP & speech)
    ESPnet (speech recognition, translation, synthesis, ...)
"""

"""
    有用的网站
    https://pytorch.org/
    https://github.com/pytorch/pytorch
    https://github.com/wkentaro/pytorch-for-numpy-users
    https://blog.udacity,com/2020/05/pytorch-vs-tensorflow-what-you-need-to-know.html
    https://www.tensorflow.org/
    nttps://numpy.org/
"""


"""
batch_size: 小batch收敛慢，但是更容易到更好的minimal位置
learning_rate:
norm of gradient: 画图看是否下降
cross-entropy公式： -(sumation of y*ln(y_hat))
batch_normalization: 直接修改error surface，方便梯度下降， 在每个batch上做feature_normalization；
    放在激活函数前后都可以； u和sigma是算出来的，之后的gamma和beta是需要学习的。
    在model.eval()的时候， u和sigma用的是ema算出来的
feature_normalization: 对于feature 每个维度做normalization
"""

"""
    normalization 方法很多：
    Batch Renormalization: https://arxiv.org/abs/1702.03275
    Layer Normalization: https://arxiv.org/abs/1607.06450
    Instance Normalization: https://arxiv.org/abs/1607.08022
    Group Normalization: https://arxiv.org/abs/1803.08494
    Weight Normalization: https://arxiv.org/abs/1602.07868
    Spectrum Normalization: https://arxiv.org/abs/1705.10941
"""