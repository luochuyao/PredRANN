import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import shutil
import argparse
import numpy as np
import torch
from data.CIKM.cikm_radar import *
from core.models.model_factory import Model
from utils import preprocess
from core import trainer
import cv2
import math

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRANN-s')

# training/test
parser.add_argument('--is_training', type=int, default=1)
# parser.add_argument('--device', type=str, default='gpu:0')

# data
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--save_dir', type=str, default='../model_lib/cikm_predrann_s')
parser.add_argument('--gen_frm_dir', type=str, default='/mnt/A/meteorological/2500_ref_seq/CIKM_predrann_s_test/')
parser.add_argument('--input_length', type=int, default=5)
parser.add_argument('--total_length', type=int, default=15)
parser.add_argument('--img_width', type=int, default=128)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrann_s')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)


# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=200)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
batch_size = args.batch_size


data_root = '/mnt/A/CIKM2017/CIKM_datasets/'
train_data = Radar(
    data_type='train',
    data_root=data_root,
)
valid_data = Radar(
    data_type='validation',
    data_root=data_root
)
test_data = Radar(
    data_type='test',
    data_root=data_root
)
train_loader = DataLoader(train_data,
                          num_workers=2,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=False,
                          pin_memory=False)
valid_loader = DataLoader(valid_data,
                          num_workers=2,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=False)
test_loader = DataLoader(test_data,
                         num_workers = 2,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False,
                         pin_memory=False)

def padding_CIKM_data(frame_data):
    shape = frame_data.shape
    batch_size = shape[0]
    seq_length = shape[1]
    padding_frame_dat = np.zeros((batch_size,seq_length,args.img_width,args.img_width,args.img_channel))
    padding_frame_dat[:,:,13:-14,13:-14,:] = frame_data
    return padding_frame_dat

def unpadding_CIKM_data(padding_frame_dat):
    return padding_frame_dat[:,:,13:-14,13:-14,:]



def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def wrapper_test(model,is_save=True):
    test_save_root = args.gen_frm_dir
    if not os.path.exists(test_save_root):
        os.mkdir(test_save_root)
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    output_length = args.total_length - args.input_length
    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader):

            ims = batch_data[0].numpy()
            tars = ims[:, -output_length:]
            cur_fold = batch_data[1]
            ims = padding_CIKM_data(ims)
            ims = preprocess.reshape_patch(ims, args.patch_size)
            img_gen,_ = model.test(ims, real_input_flag)
            img_gen = img_gen.transpose((0,1,3,4,2))
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            img_out = unpadding_CIKM_data(img_gen[:, -output_length:])
            mse = np.mean(np.square(img_out-tars))

            loss = loss + mse
            img_out[img_out<0]=0
            img_out[img_out>1]=1
            img_out = (img_out*255.0).astype(np.uint8)
            count = count + 1
            if is_save:
                for bat_ind in range(batch_size):
                    cur_batch_data = img_out[bat_ind,:,:,:,0]
                    cur_sample_fold = os.path.join(test_save_root,cur_fold[bat_ind])
                    if not os.path.exists(cur_sample_fold):
                        os.mkdir(cur_sample_fold)
                    for t in range(10):
                        cur_save_path = os.path.join(cur_sample_fold,'img_'+str(t+6)+'.png')
                        cur_img = cur_batch_data[t]
                        cv2.imwrite(cur_save_path, cur_img)


    print('test loss is:',str(loss/count))
    return loss / count


def wrapper_valid(model):
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))

    for i_batch, batch_data in enumerate(valid_loader):
        ims = batch_data.numpy()
        ims = padding_CIKM_data(ims)
        ims = preprocess.reshape_patch(ims, args.patch_size)
        _,mse = model.test(ims, real_input_flag)
        loss = loss+mse
        count = count+1

    return loss/count




def wrapper_train(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)

    eta = args.sampling_start_value
    best_mse = math.inf
    tolerate = 0
    limit = 3
    best_iter = None
    for itr in range(1, args.max_iterations + 1):
        for i_batch, batch_data in enumerate(train_loader):

            ims = batch_data.numpy()
            ims = padding_CIKM_data(ims)
            ims = preprocess.reshape_patch(ims, args.patch_size)
            eta, real_input_flag = schedule_sampling(eta, itr)
            cost = trainer.train(model, ims, real_input_flag, args, itr)

            if (i_batch+1) % args.display_interval == 0:
                print('itr: ' + str(itr))
                print('training loss: ' + str(cost))

        if (itr+1) % args.test_interval == 0:
            print('validation one ')
            valid_mse = wrapper_valid(model)
            print('validation mse is:',str(valid_mse))


            if valid_mse<best_mse:
                best_mse = valid_mse
                best_iter = itr
                tolerate = 0
                model.save()
            else:
                tolerate = tolerate+1

            if tolerate==limit:
                model.load()
                test_mse = wrapper_test(model)
                print('the best valid mse is:',str(best_mse))
                print('the test mse is ',str(test_mse))
                break


if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

# gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
# args.n_gpu = len(gpu_list)
# print('Initializing models')

model = Model(args)
# model.load()
print("the test loss is:",str(wrapper_test(model)))

if args.is_training:
   wrapper_train(model)
else:
   wrapper_test(model)
