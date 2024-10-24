import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import math


def build_lr_scheduler(cfg, optimizer, last_epoch):
    """
    构建学习率调度器和学习率预热调度器
    :param cfg: 配置文件中的调度器参数
    :param optimizer: 优化器，用于更新模型的权重
    :param last_epoch: 上一次训练的轮数
    :return: 返回学习率调度器和可选的预热调度器
    """
    # 定义学习率的调整函数，根据当前epoch衰减学习率
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']: # 如果当前epoch大于等于某个衰减步长，则对学习率进行多次衰减
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return cur_decay
    # 使用LambdaLR根据自定义的lr_lbmd函数调整学习率
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    warmup_lr_scheduler = None
    if cfg['warmup']:  # 如果配置启用了学习率预热
        warmup_lr_scheduler = CosineWarmupLR(optimizer, num_epoch=5, init_lr=0.00001)  # 使用余弦预热调度器
    return lr_scheduler, warmup_lr_scheduler


def build_bnm_scheduler(cfg, model, last_epoch):
    """
    构建BatchNorm层动量的调度器
    :param cfg: 配置文件中的调度器参数
    :param model: 神经网络模型
    :param last_epoch: 上一次训练的轮数
    :return: 返回BatchNorm动量调度器
    """
    if not cfg['enabled']:
        return None

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return max(cfg['momentum']*cur_decay, cfg['clip'])  # 动量值不能低于配置的剪辑值

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    """
    设置BatchNorm层的默认动量
    :param bn_momentum: 动量值
    :return: 返回用于设置动量的函数
    """
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    """
    BatchNorm层动量调度器
    """
    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        """
        初始化调度器
        :param model: 神经网络模型
        :param bn_lambda: 调整动量的函数
        :param last_epoch: 上一次训练的轮数
        :param setter: 用于设置动量的函数
        """
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    """
    余弦预热学习率调度器
    """
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        """
        初始化调度器
        :param optimizer: 优化器
        :param num_epoch: 预热阶段的总epoch数
        :param init_lr: 初始学习率
        :param last_epoch: 上一次训练的轮数
        """
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) *
                (1 - math.cos(math.pi * self.last_epoch / self.num_epoch)) / 2
                for base_lr in self.base_lrs]


class LinearWarmupLR(lr_sched._LRScheduler):
    """
    线性预热学习率调度器
    """
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        """
        初始化调度器
        :param optimizer: 优化器
        :param num_epoch: 预热阶段的总epoch数
        :param init_lr: 初始学习率
        :param last_epoch: 上一次训练的轮数
        """
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        获取当前的学习率
        :return: 计算后的学习率列表
        """
        return [self.init_lr + (base_lr - self.init_lr) * self.last_epoch / self.num_epoch
                for base_lr in self.base_lrs]



if __name__ == '__main__':
    # testing
    import torch.optim as optim
    from lib.models.centernet3d import CenterNet3D
    import matplotlib.pyplot as plt

    net = CenterNet3D()  # 初始化模型
    optimizer = optim.Adam(net.parameters(), 0.01)
    lr_warmup_scheduler_cosine = CosineWarmupLR(optimizer, 1000, init_lr=0.00001, last_epoch=-1)
    lr_warmup_scheduler_linear = LinearWarmupLR(optimizer, 1000, init_lr=0.00001, last_epoch=-1)

    batch_cosine, lr_cosine = [], []  # 用于保存余弦调度器的批次和学习率
    batch_linear, lr_linear = [], []  # 用于保存线性调度器的批次和学习率

    for i in range(1000):
        batch_cosine.append(i)
        lr_cosine.append(lr_warmup_scheduler_cosine.get_lr())  # 获取当前学习率并保存
        batch_linear.append(i)
        lr_linear.append(lr_warmup_scheduler_linear.get_lr())  # 获取当前学习率并保存
        lr_warmup_scheduler_cosine.step()
        lr_warmup_scheduler_linear.step()

    # 可视化学习率变化
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(batch_cosine, lr_cosine, c = 'r',marker = 'o')
    ax2 = fig.add_subplot(122)
    ax2.scatter(batch_linear, lr_linear, c = 'r',marker = 'o')
    plt.show()



