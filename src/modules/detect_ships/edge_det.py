import numpy as np
import os
import cv2
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import Custom_Dataset
from model import RCF
from src.logger import logging



def single_scale_test(model, test_loader, save_dir):
    model.eval()
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        all_res = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          all_res[i, 0, :, :] = results[i]
        filename = f'three_ships_horizon_{idx}'
        torchvision.utils.save_image(1 - all_res, os.path.join(save_dir, '%s.jpg' % filename))
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, '%s_ss.png' % filename), fuse_res)
    print('Running single-scale test done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    parser.add_argument('--checkpoint', default='bsds500_pascal_model.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--save-dir', help='output folder', default='results')
    parser.add_argument('--dataset', help='root folder of dataset', default='data')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
  
    test_dataset  = Custom_Dataset(root=args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    
    model = RCF().cuda()

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint from '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        print("=> checkpoint loaded")
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

    print('Performing the testing...')
    single_scale_test(model, test_loader, args.save_dir)
