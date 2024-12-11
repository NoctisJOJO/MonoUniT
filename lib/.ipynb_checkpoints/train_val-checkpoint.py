import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  #忽略用户代码产生的警告

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #/root/MonoUNI/lib
ROOT_DIR = os.path.dirname(BASE_DIR)                   #/root/MonoUNI
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.rope3d import Rope3D
import torch
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# test
def my_worker_init_fn(worker_id):
    """工作进程初始化函数，为每个DataLoader进程设置不同的随机种子"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def create_logger(log_file):
    """日志记录器"""
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main_worker(local_rank, nprocs, args):  
    """主工作进程函数，负责模型训练或评估的整个流程"""
    # load cfg  加载配置文件
    args.local_rank = local_rank  # 设置本地进程的rank
    assert(os.path.exists(args.config))  # 确保配置文件存在  assert：断言
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)  # 读取配置文件内容
    
    import shutil
    if not args.evaluate and local_rank==0:  # 如果不是评估模式，并且是主进程
        if os.path.exists(os.path.join(cfg['trainer']['log_dir'], 'lib/')):  # 检查日志目录
                                                                             # (cfg['trainer']['log_dir'] == ./output/rope3d)
                                                                             # 下是否存在lib目录，即 ./output/rope3d/lib/
            shutil.rmtree(os.path.join(cfg['trainer']['log_dir'], 'lib/'))   # 删除lib目录
        
        shutil.copytree('./lib', os.path.join(cfg['trainer']['log_dir'], 'lib/'))  # 将当前lib目录复制到日志目录内

    if args.seed is not None:
        random.seed(args.seed)  # python的随机种子
        torch.manual_seed(args.seed)  # pytorch的随机种子
        cudnn.deterministic = True  # 设置cudnn为确定模式
        #cudnn.benchmark=False      # benchmark为True时能增加网络运行速度，但是结果无法复现
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')  # 警告用户使用随机种子可能会降低训练速度，并可能导致从检查点恢复时出现意外行为

    # ip = random.randint(1000,10000)
    # 初始化分布式进程组，进行分布式训练
    dist.init_process_group(backend='nccl',  # 使用nccl后端
                        init_method='tcp://127.0.0.1:'+str(args.ip),  # 初始化方法
                        world_size=args.nprocs,  # 世界进程（GPU或设备）总数
                        rank=local_rank)  # 当前进程的rank（用于标识进程或区分不同的计算资源）

    os.makedirs(cfg['trainer']['log_dir'],exist_ok=True)  # 创建日志目录
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'],'train.log'))  # 创建日志  
    
    # 加载训练数据并创建DataLoader
    train_set = Rope3D(root_dir=cfg['dataset']['root_dir'], split='train', cfg=cfg['dataset']) # 创建训练集
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # 创建分布式采样器
    train_loader = DataLoader(dataset=train_set,
                                batch_size= int(cfg['dataset']['batch_size'] * 4 / args.nprocs),
                                num_workers=2,  # 进行数据加载的工作进程数
                                shuffle=False,  # 不进行打乱数据
                                pin_memory=True,  # 将数据加载到CUDA固定内存中
                                drop_last=False,  # 保留最后一个不满批次的数据
                                sampler=train_sampler)
    # 加载验证数据集并创建DataLoader
    val_set = Rope3D(root_dir=cfg['dataset']['root_dir'], split='val', cfg=cfg['dataset'])
    val_loader = DataLoader(dataset=val_set,
                                batch_size=cfg['dataset']['batch_size']*4,
                                num_workers=2,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
    # build model
    model = build_model(cfg['model'],train_loader.dataset.cls_mean_size)
    if args.evaluate:  # 如果是评估模式
        tester = Tester(cfg, model, val_loader, logger)  # 创建Tester对象
        tester.test()  # 执行测试
        return                                                                   


    # print(local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)  # 将模型包装为分布式数据并行模型
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)

    # print(f"model: {next(model.parameters()).device}")

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler  构建学习率调度器
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      train_sampler=train_sampler,
                      local_rank=local_rank,
                      args=args)
    trainer.train()

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='implementation of GUPNet')
    parser.add_argument('-e', '--evaluate', dest='evaluate',action='store_true',help='evaluate model on validation set')
    parser.add_argument('--config', type=str, default = 'lib/config.yaml')
    parser.add_argument('--seed',default=None,type=int, help='seed for initializing training. ')
    parser.add_argument('--local_rank',default=0,type=int,help='node rank for distributed training')
    parser.add_argument('--ip',default=1222,type=int,help='node rank for distributed training')
    

    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank,args.nprocs, args)
    # mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))  # 如果需要支持多进程并行，可以使用mp.spawn来启动多个进程
