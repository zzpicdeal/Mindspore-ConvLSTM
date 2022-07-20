# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
import os
import argparse
import ast
import os
import sys

import yaml
import mindspore
import numpy as np 
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

#from src.args import args
from src.tools.callback import EvaluateCallBack

from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import  set_device, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer
from src.configs import parser as _parser

from dataset import get_dataset
from model import ConvLSTM

environment = 'train'  
if environment == 'debug':
    workroot = '/home/ma-user/work' #调试任务使用该参数
else:
    workroot = '/home/work/user-job-dir' # 训练任务使用该参数
print('current work mode:' + environment + ', workroot:' + workroot)

parser = argparse.ArgumentParser(description='MindSpore Lenet Example')

# define 2 parameters for running on modelArts
# data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default= workroot + '/data/')

parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default= workroot + '/model/')

parser.add_argument("--batch_size", default=32, type=int, metavar="N",
                    help="mini-batch size (default: 256), this is the total "
                            "batch size of all GPUs on the current node when "
                            "using Data Parallel or Distributed Data Parallel")


parser.add_argument("--device_id", default=0, type=int, help="Device Id")
parser.add_argument("--device_num", default=1, type=int, help="device num")
parser.add_argument("--epochs", default=500, type=int, metavar="N", help="number of total epochs to run")

parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
parser.add_argument("--save_every", default=2, type=int, help="Save every ___ epochs(default:2)")


parser.add_argument("--run_modelarts", type=ast.literal_eval, default=False, help="Whether run on modelarts")
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: CPU),若要在启智平台上使用NPU，需要在启智平台训练界面上加上运行参数device_target=Ascend')


random_seed = 1996
np.random.seed(random_seed)
mindspore.set_seed(random_seed)


def main():
    print(os.listdir(workroot))
    args = parser.parse_args()
    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(args.device_num)
        os.environ["RANK_SIZE"] = str(args.device_num)   

    set_seed(args.seed)
    
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[1], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)

    # get model and cast amp_level
    net = ConvLSTM(input_dim=1,
                 hidden_dim=[64, 64,64,64],
                 kernel_size=[(3, 3),(3, 3),(3,3),(3,3)],
                 num_layers=4,
                 deco_dims = [0,0,0,0],
                 enco = True,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)#
    #cast_amp(net)
    criterion = nn.MSELoss()#
    net_with_loss = NetWithLoss(net, criterion)
    data = get_dataset(args.data_url,args.batch_size)#
    batch_num = data.train_dataset.get_dataset_size()
    min_lr = 0.00001
    max_lr = 0.001
    decay_steps = 1000
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    optimizer = nn.Adam(
                net.trainable_params(),
                learning_rate = cosine_decay_lr 
            )#



    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step( net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion)
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={'SSIM': nn.MAE(), "MSE":nn.MSE(),"MAE":nn.MAE()},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=args.save_every)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())

    ckpt_save_dir = "./ckpt_" + str(rank)
    if True:
        ckpt_save_dir = workroot + '/model/ckpt_' + str(rank)

    ckpoint_cb = ModelCheckpoint(prefix='convlstm' + str(rank), directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor(100)
    eval_cb = EvaluateCallBack(model, eval_dataset=data.val_dataset, src_url=ckpt_save_dir,
                               train_url=os.path.join(args.train_url, "ckpt_" + str(rank)),
                               save_freq=args.save_every)

    print("begin train")
    model.train(int(args.epochs), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb],
                dataset_sink_mode=True)
    print("train success")

    
    #import moxing as mox
    #mox.file.copy_parallel(src_url=ckpt_save_dir, dst_url=os.path.join(args.train_url, "ckpt_" + str(rank)))


if __name__ == '__main__':
    main()
