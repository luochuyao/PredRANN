import os
import cv2
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PaddingRadar_(Dataset):
    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))
    def __len__(self):
        return len(self.dirs)

    def padding_img(self,data):
        padding_data = np.zeros((128,128,1))
        padding_data[13:-14,13:-14,:] = data
        return padding_data

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path,0)[:,:,np.newaxis]
            img = self.padding_img(img)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)/255.0
        return imgs,self.dirs[index]


class PaddingRadar(Dataset):
    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))
    def __len__(self):
        return len(self.dirs)

    def padding_img(self,data):
        padding_data = np.zeros((128,128,1))
        padding_data[13:-14,13:-14,:] = data
        return padding_data

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path,0)[:,:,np.newaxis]
            img = self.padding_img(img)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)/255.0
        if self.data_type == 'test':
            return imgs,self.dirs[index]
        else:
            return imgs

class Radar(Dataset):
    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))
    def __len__(self):
        return len(self.dirs)

    def padding_img(self,data):
        padding_data = np.zeros((1,128,128))
        padding_data[:,13:-14,13:-14] = data
        return padding_data

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path,0)[:,:,np.newaxis]
            # img = self.padding_img(img)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)/255.0
        if self.data_type == 'test':
            return imgs,self.dirs[index]
        else:
            return imgs




if __name__ == '__main__':
    # from core.models.predict import *
    batch_size = 4
    data_root = '/mnt/A/CIKM2017/CIKM_datasets/'
    flow_root = '/mnt/A/meteorological/2500_ref_seq/vet_test/'


    import argparse
    from torch.autograd import Variable
    from torch.optim import Adam

    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default=-1,
                        help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                        help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                        help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default=255.)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true',
                        help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_visualize', action='store_true',
                        help="visualize the optical flow during inference")
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                        help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    args = parser.parse_args()

    # model = TimeFlowNet(args=args)
    # criterion = nn.MSELoss()
    # optimizer = Adam(model.parameters(), lr=0.0001)



    # batch_size = 4
    # data_root = '/mnt/A/CIKM2017/CIKM_datasets/'
    # train_data = Radar(
    #     data_type='train',
    #     data_root=data_root,
    # )
    # valid_data = Radar(
    #     data_type='validation',
    #     data_root=data_root
    # )
    # test_data = Radar(
    #     data_type='test',
    #     data_root=data_root
    # )
    # train_loader = DataLoader(train_data,
    #                           num_workers=2,
    #                           batch_size=batch_size,
    #                           shuffle=False,
    #                           drop_last=False,
    #                           pin_memory=False)
    # valid_loader = DataLoader(valid_data,
    #                           num_workers=1,
    #                           batch_size=batch_size,
    #                           shuffle=False,
    #                           drop_last=False,
    #                           pin_memory=False)
    # test_loader = DataLoader(test_data,
    #                          num_workers = 1,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          drop_last=False,
    #                          pin_memory=False)
    #
    # for i_batch,batch_data in enumerate(train_loader):
    #     # batch_data = batch_data.cuda()
    #     print('train',str(i_batch),batch_data.numpy().shape)
    # for i_batch, batch_data in enumerate(valid_loader):
    #     # batch_data = batch_data.cuda()
    #     print('valid',str(i_batch), batch_data.numpy().shape)
    # for i_batch, batch_data in enumerate(test_loader):
    #     # batch_data = batch_data.cuda()
    #     print('test',str(i_batch), batch_data.numpy().shape)
    #     # batch_data = batch_data.detach().cpu().numpy()[0,:,0,:,:]
    #     # t_length = batch_data.shape[0]
    #     # for t in range(t_length):
    #     #     cur_img = batch_data[t]
    #     #     cur_path = os.path.join(save_fold,'img_'+str(t+1)+'.png')
    #     #     imsave(cur_path,cur_img)
    #     # print(batch_data.shape,str(i_batch))

