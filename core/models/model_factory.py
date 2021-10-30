import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import predict




class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrann_s':predict.PredRANN_S,
            'predrann_t':predict.PredRANN_T,
            'predrann':predict.PredRANN,
            'predrnn':predict.PredRNN,
            # 'rap_net':predict.RAP_Net,
            # 'rap_cell':predict.RAP_Cell,
            # 'rap_cell_h':predict.RAP_Cell_h,
            # 'rap_cell_x':predict.RAP_Cell_x,
        }

        if configs.model_name in networks_map:

            Network = networks_map[configs.model_name]
            self.network = Network(configs).cuda()
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss(size_average=True)
        self.MAE_criterion = nn.L1Loss(size_average=True)


    def save(self,ite = None):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        if ite == None:
            checkpoint_path = os.path.join(self.configs.save_dir, 'radar_model.ckpt')
        else:
            checkpoint_path = os.path.join(self.configs.save_dir, 'radar_model_'+str(ite)+'.ckpt')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self):
        checkpoint_path = os.path.join(self.configs.save_dir, 'radar_model.ckpt')
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])
        print('model has been loaded in',checkpoint_path)

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).cuda()
        mask_tensor = torch.FloatTensor(mask).cuda()

        self.optimizer.zero_grad()
        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])\
               +0.2*self.MAE_criterion(next_frames, frames_tensor[:, 1:])

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).cuda()
        mask_tensor = torch.FloatTensor(mask).cuda()
        frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_tensor = mask_tensor.permute(0, 1, 4, 2, 3).contiguous()
        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])

        return next_frames.detach().cpu().numpy(),loss.detach().cpu().numpy()