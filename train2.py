from mindspore.train.callback import Callback
import time

from mindspore import log as logger
class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): Dataset size. Default: None.
    """

    def __init__(self, data_size=None):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            logger.error("data_size must be positive int.")
            return

        step_seconds = epoch_seconds / step_size
        with open('./time.txt','a') as f:
            f.write("epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds)+'\n')
        print("epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds), flush=True)
        
        
import mindspore
import moxing as mox
import  os
import  sys
import  time
import  glob
import  numpy as np
import  logging
import  argparse
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Tensor, Model

from model_graph import *
from model_graph import OPS,Genotype,operations
from mindspore.common import set_seed
from mindspore import context, DatasetHelper, connect_network_with_dataset
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.parallel._utils import _get_device_num

import mindspore.ops as ops
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
import moxing as mox
from dataset import get_dataset, _get_rank_info
from model import ConvLSTM
set_seed(1996)

parser = argparse.ArgumentParser("convlstm")
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
#parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')#save_path
#parser.add_argument('--test_url', required=True, default=None, help='Location of data.')
parser.add_argument('--num_parallel_workers', type=int, default=1, help='num_parallel_work')
parser.add_argument("--save_every", default=2, type=int, help="Save every ___ epochs(default:2)")
#parser.add_argument('--continue', type=bool, default=False, help='continue train')
args = parser.parse_args()
#context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
#context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
context.set_context(device_id=device_id) # set device_id
init()

random_seed = 1996
np.random.seed(random_seed)

 ######################## 将多个数据集从obs拷贝到训练镜像中 （固定写法）########################  
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    return 
 ######################## 将输出的模型拷贝到obs（固定写法）########################  
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return     
environment = 'train'  
if environment == 'debug':
    workroot = '/home/ma-user/work' #调试任务使用该参数
else:
    workroot = '/home/work/user-job-dir' # 训练任务使用该参数
print('current work mode:' + environment + ', workroot:' + workroot)


def main():
    rank =get_rank()
    print(os.listdir(workroot))
    #初始化数据和模型存放目录
    data_dir = workroot + '/data'  #先在训练镜像中定义数据集路径
    train_dir = workroot + '/model' #先在训练镜像中定义输出路径
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_dir):
            os.makedirs(train_dir)
 ######################## 将数据集从obs拷贝到训练镜像中 （固定写法）########################   
    # 在训练环境中定义data_url和train_url，并把数据从obs拷贝到相应的固定路径，以下写法是将数据拷贝到/home/work/user-job-dir/data/目录下，可修改为其他目录
    ObsToEnv(args.data_url,data_dir)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)

    net = ConvLSTM(input_dim=1,
                 hidden_dim=[64, 64,64,64],
                 kernel_size=[(3, 3),(3, 3),(3,3),(3,3)],
                 num_layers=4,
                 deco_dims = [0,0,0,0],
                 enco = True,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
    criterion = nn.MSELoss()#
    #net_with_loss = NetWithLoss(net, criterion)
    data = get_dataset(data_dir,args.batch_size)#
    batch_num = data.train_dataset.get_dataset_size()
    min_lr = 0.000001
    max_lr = 0.0001
    decay_steps = 1000
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    optimizer = nn.Adam(
                net.trainable_params(),
                learning_rate = cosine_decay_lr) 

    model = Model(network=net, loss_fn=criterion, optimizer=optimizer, metrics={'SSIM': nn.MAE(), "MSE":nn.MSE(),"MAE":nn.MAE()})
    ckpt_save_dir = workroot + '/model/ckpt_' + str(rank)
    loss_cb = LossMonitor(100)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=500, keep_checkpoint_max=16)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=ckpt_save_dir, config=config_ck)
    print(_get_device_num())
    print("begin train")
    model.train(int(args.epochs), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb],
                dataset_sink_mode=False)
    print("train success")

    EnvToObs(train_dir, args.train_url)
            