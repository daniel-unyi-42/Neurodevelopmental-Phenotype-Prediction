import torch
import torch.nn.functional as F

def train_model(train_loader, model, optimizer):
    model.train()
    train_accs = []
    for train_data in train_loader:
        optimizer.zero_grad()
        x, y_true = train_data
        x = x.to(model.device)
        y_true = y_true.to(model.device)
        y_pred = model(x)
        train_loss = F.mse_loss(y_pred, y_true)
        train_loss.backward()
        optimizer.step()
        train_acc = F.l1_loss(y_pred, y_true)
        train_accs.append(train_acc)
    return sum(train_accs) / len(train_accs)

def test_model(test_loader, model):
    model.eval()
    test_accs = []
    with torch.no_grad():
        for test_data in test_loader:
            x, y_true = test_data
            x = x.to(model.device)
            y_true = y_true.to(model.device)
            y_pred = model(x)
            test_acc = F.l1_loss(y_pred, y_true)
            test_accs.append(test_acc)
    return sum(test_accs) / len(test_accs)
