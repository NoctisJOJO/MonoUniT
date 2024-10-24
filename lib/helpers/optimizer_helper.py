import torch.optim as optim


def build_optimizer(cfg_optimizer, model):
    """
    根据配置文件构建模型的优化器
    :param cfg_optimizer: 配置文件中关于优化器的参数
    :param model: 需要优化的模型
    :return: 构建好的优化器
    """
    weights, biases = [], []
    for name, param in model.named_parameters():  # 遍历模型的所有参数
        if 'bias' in name:  # 如果参数名中包含'bias'
            biases += [param]  # 将该参数归入biaser列表
        else:
            weights += [param]  # 否则归入weights列表

    parameters = [{'params': biases, 'weight_decay': 0},  # 偏置项不进行权重衰减
                  {'params': weights, 'weight_decay': cfg_optimizer['weight_decay']}]

    if cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer