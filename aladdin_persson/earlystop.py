# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/17


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        连续patience轮，val_loader上没有提升min_delta的loss，就停止
        :param patience: 在多少个epoch后没有改进时停止训练
        :param min_delta: 提高的最小阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 假设有一个验证数据加载器
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

# 实例化早停对象
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在验证集上评伤断续
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            data = data.reshape(data.shape[0], -1)
            scores = model(data)
            val_loss += criterion(scores, targets).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")

    # 调用早停逻辑
    early_stopping(val_leass)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break
