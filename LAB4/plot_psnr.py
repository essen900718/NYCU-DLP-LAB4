import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from glob import glob
from torch import stack
from torch.utils.data import Dataset as torchData
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader as imgloader
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
from torchvision.utils import save_image

import imageio
from math import log10
from Trainer import VAE_Model
import matplotlib.pyplot as plt


def get_key(fp):
    filename = fp.split('\\')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

class Dataset_Dance(torchData):
    def __init__(self, root, transform, mode='valid', video_len=7, partial=1.0):
        super().__init__()
        
        self.img_folder = sorted(glob(os.path.join(root, 'val\\val_img\\*.png')), key=get_key)
        self.prefix = 'val'
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]

        imgs = []
        labels = []
        for i in range(self.video_len):
            label_list = self.img_folder[(index*self.video_len)+i].split('\\')
            label_list[-2] = self.prefix + '_label'
            
            img_name   = self.img_folder[(index*self.video_len)+i]
            label_name = '\\'.join(label_list)

            imgs.append(self.transform(imgloader(img_name)))
            labels.append(self.transform(imgloader(label_name)))
        
        return stack(imgs), stack(labels)
    
"""PSNR for torch tensor"""
def Generate_PSNR(imgs1, imgs2, data_range=1.):
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

class Test_model(VAE_Model):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size
        
    def forward(self, img, label):
        reconstructed_img = [img[:,0]]    # Initialize frame
        
        for i in range(1, img.size(1)):
            previous_frames = reconstructed_img[-1]
            
            # Image transformation
            previous_features = self.frame_transformation(previous_frames)
            ground_truth_features = self.frame_transformation(img[:, i])
        
            # Label transformation
            label_features = self.label_transformation(label[:, i])
        
            # Predict latent parameters (sample latent variable)
            z, mu, logvar = self.Gaussian_Predictor(ground_truth_features, label_features)

            # Fusion for decoder input
            decoder_input = self.Decoder_Fusion(previous_features, label_features, z)

            # Generate output
            generated_output = self.Generator(decoder_input)
            reconstructed_img.append(generated_output)
        
        reconstructed_img = stack(reconstructed_img, dim = 1)
        
        return reconstructed_img, mu, logvar    
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        psnr_list = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            generated_frame = self.val_one_step(img, label)

            for i in range(1, len(generated_frame)):
                PSNR = Generate_PSNR(img[0][i], generated_frame[i])
                psnr_list.append(PSNR.item())

        return psnr_list, sum(psnr_list)/(len(psnr_list)-1)

    def val_one_step(self, img, label):
        # Forward pass
        reconstructed_img, mu, logvar = self.forward(img, label)

        return reconstructed_img.squeeze()
                
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, video_len=self.val_vi_len) 
        val_loader = DataLoader(dataset, batch_size=1, num_workers=self.args.num_workers,
                                drop_last=True, shuffle=False)  
        return val_loader

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
    
def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = Test_model(args).to(args.device)
    model.load_checkpoint()
    PSNR_perFrame, AVG_psnr = model.eval()
    AVG_psnr = round(AVG_psnr, 3)
    
    graph = []
    for i in range(629):
        graph.append(i+1)

    graph = np.array(graph)
    fig = plt.figure()
    plt.title('Per frame Quality(PSNR)')
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("PSNR", fontsize=12)
    plt.plot(graph, PSNR_perFrame, label=f'AVG_PSNR: {AVG_psnr}')
    plt.legend()
    plt.savefig(args.save_root + '\\' + 'PSNR-per frame diagram.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--no_sanity',     action='store_true')
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--make_gif',      action='store_true')
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,   default=None, help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Monotonic Cyclical
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    args = parser.parse_args()
    
    main(args)





