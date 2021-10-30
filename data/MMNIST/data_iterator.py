import random
import numpy as np
from scipy.misc import *

import os
import cv2
# from scipy.misc import imread
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MNIST_HALF(Dataset):
    def __init__(self,data_type,data_root='/mnt/A/MNIST_dataset/'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test_set1, test_set2, test_set3, test_set4]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))


    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            if i % 2 == 0:
                file = 'img_'+str(i+1)+'.png'
                img_path = os.path.join(cur_fold,file)
                img = cv2.imread(img_path,0)[:,:,np.newaxis]
                imgs.append(img)
            else:
                continue
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)/255.0
        if self.data_type[:4] == 'test':
            return imgs,self.dirs[index]
        else:
            return imgs

class MNIST(Dataset):

    def __init__(self,data_type,data_root='/mnt/A/MNIST_dataset/'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test_set1, test_set2, test_set3, test_set4]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))


    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path,0)[:,:,np.newaxis]
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)/255.0
        if self.data_type[:4] == 'test':
            return imgs,self.dirs[index]
        else:
            return imgs



def sample(batch_size,mode = 'random',data_type='train',index = None):
    save_root = '/mnt/A/MNIST_dataset/' + data_type + '/'
    if data_type == 'train':
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,8000)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []
                for t in range(1,16):
                    img_path = img_fold + 'img_'+str(t)+'.png'
                    img = cv2.imread(img_path,0)[:,:,np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = cv2.imread(img_path,0)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type[:4] == 'test':
        if index == None:
            raise('index need be initialize')
        if index>4001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>4001:
                index = 4001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = cv2.imread(img_path,0)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==4001:
            return imgs,(index,0)
        return imgs,(index,b_cup)

    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>2001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>2001:
                index = 2001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = cv2.imread(img_path,0)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==2001:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")


if __name__ == '__main__':
    batch_size = 4
    data_root = '/mnt/A/MNIST_dataset/'
    train_data = MNIST(
        data_type='train',
        data_root=data_root,
    )
    valid_data = MNIST(
        data_type='validation',
        data_root=data_root
    )
    test_data = MNIST(
        data_type='test_set1',
        data_root=data_root
    )
    train_loader = DataLoader(train_data,
                              num_workers=2,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=False)
    valid_loader = DataLoader(valid_data,
                              num_workers=1,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=False)
    test_loader = DataLoader(test_data,
                             num_workers=1,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=False)

    for i_batch, batch_data in enumerate(train_loader):
        # batch_data = batch_data.cuda()
        print('train', str(i_batch), batch_data.numpy().shape)
    for i_batch, batch_data in enumerate(valid_loader):
        # batch_data = batch_data.cuda()
        print('valid', str(i_batch), batch_data.numpy().shape)
    for i_batch, batch_data in enumerate(test_loader):
        # batch_data = batch_data.cuda()
        print('test', str(i_batch), batch_data[0].numpy().shape)
    pass
    # train_dat = sample(4,data_type = 'train')
    # print(train_dat.shape,np.max(train_dat),np.min(train_dat))
    # validation_dat,(index, b_cup) = sample(4, data_type='validation',index=1)
    # print(validation_dat.shape,np.max(validation_dat),np.min(validation_dat))
    # test_dat1,(index, b_cup)  = sample(4, data_type='test_set1',index=index)
    # print(test_dat1.shape,np.max(test_dat1),np.min(test_dat1))
    # test_dat2,(index, b_cup) = sample(4, data_type='test_set2',index=index)
    # print(test_dat2.shape,np.max(test_dat2),np.min(test_dat2))
    # test_dat3 ,(index, b_cup)= sample(4, data_type='test_set3',index=index)
    # print(test_dat3.shape,np.max(test_dat3),np.min(test_dat3))
    # test_dat4,(index, b_cup) = sample(4, data_type='test_set4',index=index)
    # print(test_dat4.shape,np.max(test_dat4),np.min(test_dat4))
