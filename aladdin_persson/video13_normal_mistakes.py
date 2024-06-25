# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/20


"""
Mistake1: didn't overfit a single batch
    在训练前先测试一个batch的forward流程，然后batch_size=1测试是否可以逼近target
    batch_data, targets = next(iter(train_loader))
Mistake2: forget to set train or eval
    model.eval(), model.train(), with torch.no_grad()
    with torch.no_grad(): 节省内存，加速推理
Mistake3: forget to zero_grad()
    optimizer.zero_grad()
Mistake4: softmax with Cross Entropy
    Cross Entropy里面已经包含了softmax的步骤，不要重复
Mistake5: use bias when using BatchNorm
    BN之前的网络，可以不用bias
Mistake6: use view(reshape) as permute
    x = torch.tensor([[1,2,3],[4,5,6]])
    x.view(3,2)和x.permute(1,0)结果是不一样的，后者才是transpose
Mistake7: incorrect data augumentation
    transforms.RandomVerticalFlip(p=1) not good
Mistake8: not shuffle data
    test_loader 可以不shuffle， 有写seq数据是不能shuffle的
Mistake9: not Normalize data
    预处理数据
Mistake10: not Clipping Gradients
    对于默写激活函数例如sigmoid等，特别在rnn， gru， lstm等 防止梯度消失
    loss.backward()之后
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm = 1)
"""

""" 常见的seed固定方法
seed=0
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
"""