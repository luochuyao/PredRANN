import torch
import torch.nn as nn
from core.layers.TransformerCell import *
from core.layers.SpatioTemporalLSTMCell import *

class PredRNN(nn.Module):
    def __init__(self, configs):
        super(PredRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames, mask_true):


        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        # memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()
        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=1)

        return next_frames

class PredRANN(nn.Module):
    def __init__(self,args):
        super(PredRANN, self).__init__()
        self.batch_size = args.batch_size
        self.input_length = args.input_length
        self.output_length = args.total_length-args.input_length
        self.width = args.img_width//args.patch_size
        self.rnn1 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )

        self.rnn2 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.rnn3 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.rnn4 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.trans_cell1 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.trans_cell2 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.trans_cell3 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.trans_cell4 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.transfs = nn.ModuleList([self.trans_cell1,self.trans_cell2,self.trans_cell3,self.trans_cell4])
        self.rnns = nn.ModuleList([self.rnn1,self.rnn2,self.rnn3,self.rnn4])
        self.spatial_trans_cell = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.hiddens = {}
        self.temporal_memory = {}
        # self.merge_layer = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.input_layer = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs, mask):
        for layer_index in range(4):
            self.hiddens[layer_index] = []
            self.hiddens[layer_index].append(torch.zeros(self.batch_size, 64, self.width, self.width).cuda())
            self.temporal_memory[layer_index] = []
            self.temporal_memory[layer_index].append(torch.zeros(self.batch_size, 64, self.width, self.width).cuda())
        spatial_memory = torch.zeros(self.batch_size, 64, self.width, self.width).cuda()
        outputs = []
        test_inputs = []
        for t in range(self.output_length+self.input_length-1):

            if t < self.input_length:
                cur_input = inputs[:, t]
            else:
                cur_input = mask[:, t - self.input_length] * inputs[:, t] + \
                            (1 - mask[:, t - self.input_length]) * cur_output

            cur_input = self.input_layer(cur_input)
            # warming stage
            test_inputs.append(cur_input)
            if t<self.input_length:
                hidden_list = []
                for layer_index in range(4):
                    if layer_index == 0:
                        hidden_list.append(cur_input)
                        hidden, temp_memory, spatial_memory = self.rnns[layer_index](cur_input,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                    else:
                        hidden_list.append(hidden)
                        hidden, temp_memory, spatial_memory = self.rnns[layer_index](hidden,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                    self.hiddens[layer_index].append(hidden)
                    self.temporal_memory[layer_index].append(temp_memory)
                spatial_hidden = self.spatial_trans_cell(hidden, hidden_list, hidden_list)
                cur_output = self.output_layer(spatial_hidden)
                outputs.append(cur_output)
            # global attention stage
            else:
                hidden_list = []
                for layer_index in range(4):
                    if layer_index == 0:
                        hidden_list.append(cur_input)
                        keys = torch.stack(test_inputs, 1)
                        values = torch.stack(test_inputs, 1)

                        hidden, temp_memory,spatial_memory = self.rnns[layer_index](cur_input,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                        hidden_ = self.transfs[layer_index](hidden,keys,values)

                    else:
                        hidden_list.append(hidden_)
                        keys = self.hiddens[layer_index-1][:t]
                        values = self.hiddens[layer_index-1][:t]

                        hidden, temp_memory,spatial_memory = self.rnns[layer_index](hidden_, self.hiddens[layer_index][t],
                                                                     self.temporal_memory[layer_index][t],spatial_memory)
                        hidden_ = self.transfs[layer_index](hidden, keys, values)

                    self.hiddens[layer_index].append(hidden)
                    self.temporal_memory[layer_index].append(temp_memory)
                spatial_hidden = self.spatial_trans_cell(hidden_, hidden_list, hidden_list)
                # hidden_ = self.merge_layer(torch.cat([spatial_hidden,hidden_],1))

                cur_output = self.output_layer(spatial_hidden)
                outputs.append(cur_output)
        outputs = torch.stack(outputs,1)
        return outputs

class PredRANN_T(nn.Module):
    def __init__(self,args):
        super(PredRANN_T, self).__init__()
        self.batch_size = args.batch_size
        self.input_length = args.input_length
        self.output_length = args.total_length-args.input_length
        self.width = args.img_width//args.patch_size
        self.rnn1 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )

        self.rnn2 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.rnn3 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.rnn4 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.trans_cell1 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.trans_cell2 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.trans_cell3 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.trans_cell4 = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.transfs = nn.ModuleList([self.trans_cell1,self.trans_cell2,self.trans_cell3,self.trans_cell4])
        self.rnns = nn.ModuleList([self.rnn1,self.rnn2,self.rnn3,self.rnn4])

        self.hiddens = {}
        self.temporal_memory = {}
        # self.merge_layer = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.input_layer = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs, mask):
        for layer_index in range(4):
            self.hiddens[layer_index] = []
            self.hiddens[layer_index].append(torch.zeros(self.batch_size, 64, self.width, self.width).cuda())
            self.temporal_memory[layer_index] = []
            self.temporal_memory[layer_index].append(torch.zeros(self.batch_size, 64, self.width, self.width).cuda())
        spatial_memory = torch.zeros(self.batch_size, 64, self.width, self.width).cuda()
        outputs = []
        test_inputs = []
        for t in range(self.output_length+self.input_length-1):

            if t < self.input_length:
                cur_input = inputs[:, t]
            else:
                cur_input = mask[:, t - self.input_length] * inputs[:, t] + \
                            (1 - mask[:, t - self.input_length]) * cur_output

            cur_input = self.input_layer(cur_input)
            # warming stage
            test_inputs.append(cur_input)
            if t<self.input_length:
                hidden_list = []
                for layer_index in range(4):
                    if layer_index == 0:
                        hidden_list.append(cur_input)
                        hidden, temp_memory, spatial_memory = self.rnns[layer_index](cur_input,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                    else:
                        hidden_list.append(hidden)
                        hidden, temp_memory, spatial_memory = self.rnns[layer_index](hidden,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                    self.hiddens[layer_index].append(hidden)
                    self.temporal_memory[layer_index].append(temp_memory)

                cur_output = self.output_layer(hidden)
                outputs.append(cur_output)
            # global attention stage
            else:
                hidden_list = []
                for layer_index in range(4):
                    if layer_index == 0:
                        hidden_list.append(cur_input)
                        keys = torch.stack(test_inputs, 1)
                        values = torch.stack(test_inputs, 1)

                        hidden, temp_memory,spatial_memory = self.rnns[layer_index](cur_input,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                        hidden_ = self.transfs[layer_index](hidden,keys,values)

                    else:
                        hidden_list.append(hidden_)
                        keys = self.hiddens[layer_index-1][:t]
                        values = self.hiddens[layer_index-1][:t]

                        hidden, temp_memory,spatial_memory = self.rnns[layer_index](hidden_, self.hiddens[layer_index][t],
                                                                     self.temporal_memory[layer_index][t],spatial_memory)
                        hidden_ = self.transfs[layer_index](hidden, keys, values)

                    self.hiddens[layer_index].append(hidden)
                    self.temporal_memory[layer_index].append(temp_memory)
                cur_output = self.output_layer(hidden_)
                outputs.append(cur_output)
        outputs = torch.stack(outputs,1)
        return outputs

class PredRANN_S(nn.Module):
    def __init__(self,args):
        super(PredRANN_S, self).__init__()
        self.batch_size = args.batch_size
        self.input_length = args.input_length
        self.output_length = args.total_length-args.input_length
        self.width = args.img_width//args.patch_size

        self.rnn1 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )

        self.rnn2 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.rnn3 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )
        self.rnn4 = SpatioTemporalLSTMCell(
            in_channel=64,
            num_hidden=64,
            width=self.width,
            filter_size=3,
            stride=1,
            layer_norm=True
        )

        self.rnns = nn.ModuleList([self.rnn1,self.rnn2,self.rnn3,self.rnn4])
        self.spatial_trans_cell = TransformerCell(
            qin_channels=64,
            kvin_channels=64,
            heads=8,
            head_channels=32,
            width=self.width)
        self.hiddens = {}
        self.temporal_memory = {}
        # self.merge_layer = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.input_layer = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs, mask):
        for layer_index in range(4):
            self.hiddens[layer_index] = []
            self.hiddens[layer_index].append(torch.zeros(self.batch_size, 64, self.width, self.width).cuda())
            self.temporal_memory[layer_index] = []
            self.temporal_memory[layer_index].append(torch.zeros(self.batch_size, 64, self.width, self.width).cuda())
        spatial_memory = torch.zeros(self.batch_size, 64, self.width, self.width).cuda()
        outputs = []
        test_inputs = []
        for t in range(self.output_length+self.input_length-1):

            if t < self.input_length:
                cur_input = inputs[:, t]
            else:
                cur_input = mask[:, t - self.input_length] * inputs[:, t] + \
                            (1 - mask[:, t - self.input_length]) * cur_output

            cur_input = self.input_layer(cur_input)
            # warming stage
            test_inputs.append(cur_input)
            if t<self.input_length:
                hidden_list = []
                for layer_index in range(4):
                    if layer_index == 0:
                        hidden_list.append(cur_input)
                        hidden, temp_memory, spatial_memory = self.rnns[layer_index](cur_input,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                    else:
                        hidden_list.append(hidden)
                        hidden, temp_memory, spatial_memory = self.rnns[layer_index](hidden,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)
                    self.hiddens[layer_index].append(hidden)
                    self.temporal_memory[layer_index].append(temp_memory)
                spatial_hidden = self.spatial_trans_cell(hidden, hidden_list, hidden_list)
                cur_output = self.output_layer(spatial_hidden)
                outputs.append(cur_output)
            # global attention stage
            else:
                hidden_list = []
                for layer_index in range(4):
                    if layer_index == 0:
                        hidden_list.append(cur_input)
                        hidden, temp_memory,spatial_memory = self.rnns[layer_index](cur_input,self.hiddens[layer_index][t],self.temporal_memory[layer_index][t],spatial_memory)


                    else:
                        hidden_list.append(hidden)
                        hidden, temp_memory,spatial_memory = self.rnns[layer_index](hidden, self.hiddens[layer_index][t],
                                                                     self.temporal_memory[layer_index][t],spatial_memory)


                    self.hiddens[layer_index].append(hidden)
                    self.temporal_memory[layer_index].append(temp_memory)
                spatial_hidden = self.spatial_trans_cell(hidden, hidden_list, hidden_list)


                cur_output = self.output_layer(spatial_hidden)
                outputs.append(cur_output)
        outputs = torch.stack(outputs,1)
        return outputs