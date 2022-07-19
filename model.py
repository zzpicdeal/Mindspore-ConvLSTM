import mindspore
import mindspore.nn as nn
import mindspore.ops as ops 
from mindspore import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Parameter
class ConvLSTMCell(nn.Cell):
    def __init__(self,input_dim,hidden_dim,kernel_size,deco_dim,bias,enco=True):
        super(ConvLSTMCell,self).__init__()
        self.input_dim = input_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2#list?
        self.bias = bias 
        if enco:
            self.hidden_dim = hidden_dim
            self.conv = nn.Conv2d(self.input_dim+self.hidden_dim,4*self.hidden_dim,
                self.kernel_size,pad_mode='pad',padding=self.padding,has_bias=self.bias)
        else:
            self.hidden_dim = deco_dim
            self.conv = nn.Conv2d(self.input_dim,self.hidden_dim//4,
                self.kernel_size,pad_mode='pad',padding=self.padding,has_bias=self.bias)
        
        self.op = ops.Concat(1)
        self.split = ops.Split(1,4)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.zeros = ops.Zeros()
    def construct(self,input_tensor,cur_state):
        h_cur , c_cur = cur_state
        combined = self.op((input_tensor,h_cur))
        
        combined_conv = self.conv(combined)
        cc_i,cc_f,cc_o,cc_g = self.split(combined_conv)
        i = self.sig(cc_i)
        f = self.sig(cc_f)
        o = self.sig(cc_o)
        g = self.tanh(cc_g)

        c_next = f*c_cur + i*g
        h_next = o*self.tanh(c_next)

        return h_next,c_next
    def init_hidden(self,batch_size,img_size):
        h,w = img_size
        return (self.zeros((batch_size,self.hidden_dim,h,w), mindspore.float32),
                self.zeros((batch_size,self.hidden_dim,h,w), mindspore.float32))
class ConvLSTM(nn.Cell):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,deco_dims=0,enco=True,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers  
        self.enco = enco
        self.deco_dim = deco_dims
        self.last = nn.Conv2d(64,1,(1,1))
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          deco_dim = self.deco_dim[i],
                                          bias=self.bias,
                                          enco=self.enco))

        self.cell_list = nn.CellList(cell_list)

        self.trans = ops.Transpose()   
        self.stack = ops.Stack(axis=1)

    def construct(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = self.trans(input_tensor,(1, 0, 2, 3, 4))

        b, _, _, h, w = input_tensor.shape

        # Implement stateful ConvLSTM
        #if hidden_state is not None:
            #raise NotImplementedError()
        #else:
            # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor
        
        #for layer_idx in range(self.num_layers):
        for layer_idx in range(2):
            h, c = hidden_state[layer_idx]
            output_inner = []
            output_inner2 = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                output_inner2.append(c)
            layer_output = self.stack(output_inner)
            cur_layer_input = layer_output
            #layer_output = self.stack(output_inner)
            #layer_output2 = self.stack(output_inner2)
        for layer_idx in range(2):
            h, c = output_inner[-layer_idx-1],output_inner2[-layer_idx-1]
            output_inner3 = []
            output_inner4 = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx+2](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner3.append(h)
                output_inner4.append(c)
            layer_output = self.stack(output_inner3)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        layer_output_list, last_state_list
        out = layer_output_list[0]
        #print(out.shape)
        B,S,N,H,W = out.shape
        out = out.reshape(B*S,N,H,W)
        out = self.last(out)
        out = out.reshape(B,S,1,H,W)
        return out

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states