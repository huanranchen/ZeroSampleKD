import torch
from torch import nn
from tqdm import tqdm


class Solver():
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 eval_loader,
                 lr=0.1,
                 weight_decay=1e-4,
                 ):
        '''

        :param train_loader:
        :param eval_loader: which to evaluate
        :param lr:
        :param weight_decay:
        '''

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        # if os.path.exists('model.pth'):
        #     self.model.load_model()
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay,
                                         momentum=0.9, nesterov=True)

    def train(self,
              total_epoch=3,
              label_smoothing=0.0,
              fp16=True,
              ):
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        prev_loss = 999
        train_loss = 99
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(self.device)
        for epoch in range(1, total_epoch + 1):
            self.model.train()
            self.warm_up(epoch + 10, now_loss=train_loss, prev_loss=prev_loss)
            prev_loss = train_loss
            train_loss = 0
            train_acc = 0
            step = 0
            pbar = tqdm(self.train_loader)

            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                if fp16:
                    with autocast():
                        x = self.model(x)  # N, 60
                        _, pre = torch.max(x, dim=1)
                        loss = criterion(x, y)
                else:
                    x = self.model(x)  # N, 60
                    _, pre = torch.max(x, dim=1)
                    loss = criterion(x, y)

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()
                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                step += 1
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss = {train_loss / step}, acc = {train_acc / step}')

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            print(f'epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}')
            torch.save(self.model.state_dict(), 'resnet50.pth')

    def warm_up(self, epoch, now_loss=None, prev_loss=None):
        if epoch <= 10:
            self.optimizer.param_groups[0]['lr'] = self.lr * epoch / 10
        elif now_loss is not None and prev_loss is not None:
            delta = prev_loss - now_loss
            if delta / now_loss < 0.02 and delta < 0.03:
                self.optimizer.param_groups[0]['lr'] *= 0.9

        p_lr = self.optimizer.param_groups[0]['lr']
        print(f'lr = {p_lr}')

    def test_acc(self):
        # TODO: test accuracy on eval loader
        pass


if __name__ == '__main__':
    from torchvision import models

    a = models.resnet50(num_classes=10)
    from data import get_CIFAR10_train, get_CIFAR10_test

    train_loader = get_CIFAR10_train(augment=False)
    test_loader = get_CIFAR10_test()

    w = Solver(a, train_loader, test_loader)
    w.train(total_epoch=100)