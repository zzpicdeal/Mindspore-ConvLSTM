import mindspore
import mindspore.nn as nn
import mindspore.ops as ops 
from mindspore import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Parameter
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops 
from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.common.initializer import One, Normal,Zero
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
class ConvLSTMCell(nn.Cell):
    def __init__(self,shape,input_dim,hidden_dim,kernel_size):
        super(ConvLSTMCell,self).__init__()
        self.input_dim = input_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2#list?


        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(self.input_dim+self.hidden_dim,4*self.hidden_dim,
            self.kernel_size,stride=1,pad_mode='pad',padding=self.padding)
        self.gro= nn.GroupNorm(4 * self.hidden_dim // 32, 4 * self.hidden_dim)
        self.shape = shape
        self.op = ops.Concat(1)
        self.split = ops.Split(1,4)
        '''        
        self.output_inner = Parameter(Tensor(shape=(10,32,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        self.hx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.cx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))

        self.c_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.h_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        '''
        #self.output_inner = Parameter(Tensor(shape=(10,4,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        #self.hx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.cx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())

        #self.c_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.h_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.input = Tensor(shape=(10,4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.zeros = ops.Zeros()
        self.stack = ops.Stack()
    def construct(self,input_tensors,cur_state,seq_len=10):

        if cur_state is None:
            h_cur , c_cur = self.zeros((32, self.hidden_dim,self.shape[0],self.shape[1]),mstype.float32), self.zeros((32,   self.hidden_dim,self.shape[0],self.shape[1]),mstype.float32)
        else:

            h_cur , c_cur = cur_state
        if input_tensors is None:
            input_tensors = self.zeros((10,32, self.input_dim,self.shape[0],self.shape[1]),mstype.float32)
        #print(input_tensor.shape,h_cur)
        output_inner = []#self.output_inner

        for index in range(seq_len):
            #print(index)
            input_tensor = input_tensors[index,...]
            #print(input_tensor.shape,h_cur.shape)
            combined = self.op((input_tensor,h_cur))
           
            #print(combined.shape,h_cur.shape)
            combined_conv = self.conv(combined)
            combined_conv = self.gro(combined_conv)
            cc_i,cc_f,cc_o,cc_g = self.split(combined_conv)
            i = self.sig(cc_i)
            f = self.sig(cc_f)
            o = self.sig(cc_o)
            g = self.tanh(cc_g)

            c_next = f*c_cur + i*g
            h_next = o*self.tanh(c_next)
            h_cur = h_next
            c_cur = c_next
            output_inner.append(h_next)
            #print(output_inner.shape,h_next.shape,c_next.shape)
        return self.stack(output_inner),(h_next,c_next)
    
class G_ConvLSTMCell(nn.Cell):
    def __init__(self,shape,input_dim,hidden_dim,kernel_size):
        super(G_ConvLSTMCell,self).__init__()
        self.input_dim = input_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2#list?


        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(self.input_dim+self.hidden_dim,4*self.hidden_dim,
            self.kernel_size,stride=1,pad_mode='pad',padding=self.padding)
        self.gro= nn.GroupNorm(4 * self.hidden_dim // 32, 4 * self.hidden_dim)
        self.shape = shape
        self.op = ops.Concat(1)
        self.split = ops.Split(1,4)
        '''        
        self.output_inner = Parameter(Tensor(shape=(10,32,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        self.hx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.cx = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))

        self.c_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        self.h_next = Parameter(Tensor(shape=(32, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
        '''
        #self.output_inner = Parameter(Tensor(shape=(10,4,hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero()))
       
        #self.hx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.cx = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())

        #self.c_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.h_next = Tensor(shape=(4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        #self.input = Tensor(shape=(10,4, hidden_dim,shape[0],shape[1]), dtype=mstype.float32, init=Zero())
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.zeros = ops.Zeros()
        self.stack = ops.Stack()
    def construct(self,input_tensors,cur_state,seq_len=10):



        h_cur , c_cur = cur_state

        output_inner = []

        for index in range(seq_len):
           
            input_tensor = input_tensors[index,...]
            #print(input_tensor.shape,h_cur.shape)
            combined = self.op((input_tensor,h_cur))
           
            #print(combined.shape,h_cur.shape)
            combined_conv = self.conv(combined)
            combined_conv = self.gro(combined_conv)
            cc_i,cc_f,cc_o,cc_g = self.split(combined_conv)
            i = self.sig(cc_i)
            f = self.sig(cc_f)
            o = self.sig(cc_o)
            g = self.tanh(cc_g)

            c_next = f*c_cur + i*g
            h_next = o*self.tanh(c_next)
            h_cur = h_next
            c_cur = c_next
            output_inner.append(h_cur)
            #print(output_inner.shape,h_next.shape,c_next.shape)
        return self.stack(output_inner),(h_cur,c_cur)
from mindspore import nn
from collections import OrderedDict


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.Conv2dTranspose(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 pad_mode='pad',
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU()))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(alpha=0.2)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               pad_mode='pad',
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU()))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(alpha=0.2)))
        print(layers)
    return nn.SequentialCell(OrderedDict(layers))    

class G_convlstm(nn.Cell):
    def __init__(self,G_ConvLSTMCell,batch):
        super(G_convlstm,self).__init__()
        self.leak_relu = nn.LeakyReLU(alpha=0.2)
        self.en_conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,pad_mode='pad',padding=1)
        self.en_conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,pad_mode='pad',padding=1)
        self.en_conv3 = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=2,pad_mode='pad',padding=1)
        

        self.en_rnn1 = G_ConvLSTMCell(shape=(64,64), input_dim=16, kernel_size=(5,5), hidden_dim =64)
        self.en_rnn2 = G_ConvLSTMCell(shape=(32,32), input_dim=64, kernel_size=(5,5), hidden_dim =96)
        self.en_rnn3 = G_ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96)
        
        
        self.dn_conv1 = nn.Conv2dTranspose(in_channels=96,out_channels=96,kernel_size=4,stride=2,pad_mode='pad',padding=1)
        self.dn_conv2 = nn.Conv2dTranspose(in_channels=96,out_channels=96,kernel_size=4,stride=2,pad_mode='pad',padding=1)
        self.dn_conv3 = nn.Conv2d(in_channels=64,out_channels=16,kernel_size=3,stride=1,pad_mode='pad',padding=1)
        self.dn_conv4 = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,stride=1,pad_mode='pad',padding=1)        

        self.dn_rnn1 = G_ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96)
        self.dn_rnn2 = G_ConvLSTMCell(shape=(32,32), input_dim=96, kernel_size=(5,5), hidden_dim =96)
        self.dn_rnn3 = G_ConvLSTMCell(shape=(64,64), input_dim=96, kernel_size=(5,5), hidden_dim =64)
        
        self.h0 = Tensor(shape=(batch,64,64,64), dtype=mstype.float32, init=Zero())
        self.c0 = Tensor(shape=(batch,64,64,64), dtype=mstype.float32, init=Zero())
        
        self.h1 = Tensor(shape=(batch,96,32,32), dtype=mstype.float32, init=Zero())
        self.c1 = Tensor(shape=(batch,96,32,32), dtype=mstype.float32, init=Zero())
        
        self.h2 = Tensor(shape=(batch,96,16,16), dtype=mstype.float32, init=Zero())
        self.c2 = Tensor(shape=(batch,96,16,16), dtype=mstype.float32, init=Zero())
        
        self.input0 = Tensor(shape=(10,batch,96,16,16), dtype=mstype.float32, init=Zero())
        self.reshape = ops.Reshape()
    def construct(self, inputs):
        inputs = inputs.transpose(1,0,2,3,4)
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.en_conv1(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        h_0 = self.h0
        c_0 = self.c0

        inputs, state_stage1 = self.en_rnn1(inputs,(h_0,c_0),seq_len=10)
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.en_conv2(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        h_0 = self.h1
        c_0 = self.c1
        
        inputs, state_stage2 = self.en_rnn2(inputs,(h_0,c_0))
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.en_conv3(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        h_0 = self.h2
        c_0 = self.c2
        
        inputs, state_stage3 = self.en_rnn3(inputs,(h_0,c_0))
        
        
        inputs, _ = self.dn_rnn1(self.input0, state_stage3, seq_len=10)
       
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.dn_conv1(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        
        
        inputs, _ = self.dn_rnn2(inputs, state_stage2, seq_len=10)
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.dn_conv2(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        
        
        inputs, _ = self.dn_rnn3(inputs, state_stage1, seq_len=10)
        
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = self.dn_conv3(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.dn_conv4(inputs)
        inputs = self.leak_relu(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        inputs = inputs.transpose(1, 0,2,3,4)
        return inputs
        
class Encoder(nn.Cell):
    def __init__(self, subnets, rnns):
        super(Encoder,self).__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)
        self.reshape = ops.Reshape()
        self.stages = [self.stage1,self.stage2,self.stage3]
        self.rnns = [self.rnn1,self.rnn2,self.rnn3]
    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def construct(self, inputs):

        inputs = inputs.transpose(1,0,2,3,4)  # to S,B,1,64,64
        hidden_states = []
        #logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            #print(i)
            inputs, state_stage = self.forward_by_stage(
                inputs, self.stages[i-1],self.rnns[i-1])
            hidden_states.append(state_stage)
        return hidden_states
class Decoder(nn.Cell):
    def __init__(self, subnets, rnns):
        super(Decoder,self).__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))
            print(self.blocks - index,rnn)
        self.reshape = ops.Reshape()
        self.stages = [self.stage1,self.stage2,self.stage3]
        self.rnns = [self.rnn1,self.rnn2,self.rnn3]
    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = self.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = self.reshape(inputs, (seq_number, batch_size, inputs.shape[1],
                                        inputs.shape[2], inputs.shape[3]))
        return inputs

        # input: 5D S*B*C*H*W

    def construct(self, hidden_states):
       
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       self.stages[2],
                                        self.rnns[2])
        #print('x',inputs.shape)
        #youcuo list 
        for i in range(0,2):
            inputs = self.forward_by_stage(inputs, hidden_states[1-i],
                                           self.stages[1-i],
                                           self.rnns[1-i])
        #inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        inputs = inputs.transpose(1, 0,2,3,4)  # to B,S,1,64,64
        return inputs


class ED(nn.Cell):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output

encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        ConvLSTMCell(shape=(64,64), input_dim=16, kernel_size=(5,5), hidden_dim =64),
        ConvLSTMCell(shape=(32,32), input_dim=64, kernel_size=(5,5), hidden_dim =96),
        ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96)
    ]
]
decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        
        ConvLSTMCell(shape=(16,16), input_dim=96, kernel_size=(5,5), hidden_dim =96),
        ConvLSTMCell(shape=(32,32), input_dim=96, kernel_size=(5,5), hidden_dim =96),
        ConvLSTMCell(shape=(64,64), input_dim=96, kernel_size=(5,5), hidden_dim =64),
    ]
]



import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import mindspore.dataset as ds

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST():
    def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''


        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root, False)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        # if self.transform is not None:
        #     images = self.transform(images)

        r = 1
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = output / 255.0
        input = input / 255.0
        # print()
        # print(input.size())
        # print(output.size())

        #out =  input,output
        return input,output

    def __len__(self):
        return self.length
def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
random_seed = 1996
np.random.seed(random_seed)

def create_train_dataset(dataset,batch=32):
    '''
      Create train dataset.
    '''

    device_num, rank_id = _get_rank_info()

    if device_num == 1 :

        train_ds = ds.GeneratorDataset(dataset, column_names=["input_images", "target_images"], shuffle=False,num_parallel_workers=8)
    else:
        train_ds = ds.GeneratorDataset(dataset, column_names=["input_images", "target_images"], shuffle=False,num_parallel_workers=8,
                                         num_shards=device_num, shard_id=rank_id)

    
    train_ds = train_ds.batch(batch, drop_remainder=True)

    return train_ds

class get_dataset():
    def __init__(self,root,batch=32) -> None:
        self.train_dataset = create_train_dataset(MovingMNIST(root,True,10,10,[3],True),batch)
        self.val_dataset = create_train_dataset(MovingMNIST(root,False,10,10,[3],True),batch)



from mindspore.train.callback import Callback
from mindspore.ops import functional as F
class lr_(Callback):
    """StopAtTime"""
    def __init__(self, fc = 0.5,times = 4):
        """init"""
        super(lr_, self).__init__()
        self.fc =  fc
        self.times = times
        self.num = 0
        self.best = 1
    def begin(self, run_context):
        """begin"""


    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        optimizer = cb_params.optimizer

        arr_lr = cb_params.optimizer.learning_rate.asnumpy()
        if loss[0].asnumpy() >= self.best:
            self.num = self.num + 1
            if self.num >= self.times :
                new_lr = arr_lr*self.fc
                if new_lr >= 1e-8:
                    F.assign(cb_params.optimizer.learning_rate, Tensor(new_lr, mstype.float32))
                    print('change lr is:',new_lr)
                self.num = 0
                
        else:
            self.best = loss[0].asnumpy()
            
            self.num = 0   
        print('loss is :',loss[0].asnumpy(),'lr is :',arr_lr) 
        
        
import mindspore
import moxing as mox
import  sys
import  time
import  glob
import  numpy as np
import  logging
import  argparse
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Tensor, Model

print(os.listdir( '/cache/code/convlstm' ))
#os.system('cd /cache/code/convlstm') 
from mindspore.common import set_seed
from mindspore import context, DatasetHelper, connect_network_with_dataset
from mindspore import dtype as mstype
from mindspore import nn,DynamicLossScaleManager
from mindspore.parallel._utils import _get_device_num

import mindspore.ops as ops
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
import moxing as mox

#from callback import EvaluateCallBack
set_seed(1996)
environment = 'train'  


workroot = '/cache/dataset'  

print('current work mode:' + environment + ', workroot:' + workroot)
parser = argparse.ArgumentParser("convlstm")
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
#parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default= workroot + '/data/')

parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default= workroot + '/model/')#save_path
#parser.add_argument('--test_url', required=True, default=None, help='Location of data.')
parser.add_argument('--num_parallel_workers', type=int, default=1, help='num_parallel_work')
parser.add_argument("--save_every", default=2, type=int, help="Save every ___ epochs(default:2)")
#parser.add_argument('--continue', type=bool, default=False, help='continue train')
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: CPU),若要在启智平台上使用NPU，需要在启智平台训练界面上加上运行参数device_target=Ascend')
args = parser.parse_args()
#context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
#context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

#device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
#context.set_context(device_id=device_id) # set device_id
#init()

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



def main():
    rank =1

    print(os.listdir(workroot))
    #初始化数据和模型存放目录
    data_dir = workroot + '/data'  #先在训练镜像中定义数据集路径
    train_dir = '/cache/output'#workroot + '/model' #先在训练镜像中定义输出路径
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_dir):
            os.makedirs(train_dir)
 ######################## 将数据集从obs拷贝到训练镜像中 （固定写法）########################   
    # 在训练环境中定义data_url和train_url，并把数据从obs拷贝到相应的固定路径，以下写法是将数据拷贝到/home/work/user-job-dir/data/目录下，可修改为其他目录
    try:
        ObsToEnv(args.data_url,data_dir)
    except:
        pass
    #context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)


    net = G_convlstm(G_ConvLSTMCell,args.batch_size)
    criterion = nn.MSELoss()#
    #net_with_loss = NetWithLoss(net, criterion)
    data = get_dataset(workroot,args.batch_size)#
    batch_num = data.train_dataset.get_dataset_size()
    max_lr = 0.0001
    min_lr = 0.000001
    decay_steps = 1000
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    optimizer = nn.Adam(
                net.trainable_params(),
                learning_rate = 0.0001) 
    
    eval_network = nn.WithEvalCell(net, criterion)
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=2**24)
    model = Model(network=net, loss_scale_manager=loss_scale_manager,loss_fn=criterion, optimizer=optimizer)
    ckpt_save_dir = workroot + '/model/ckpt_' + str(rank)
    loss_cb = LossMonitor(100)
    #time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=500, keep_checkpoint_max=16)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=ckpt_save_dir, config=config_ck)

    lr_cb = lr_(0.5,4)
    print(_get_device_num())
    print("begin train")
    model.train(int(args.epochs), data.train_dataset,
                callbacks=[ckpoint_cb,loss_cb,lr_cb],
                dataset_sink_mode=False)
    print("train success")

    try:
        EnvToObs(train_dir, args.train_url)
    except:
        pass
if __name__ == '__main__':
    main()