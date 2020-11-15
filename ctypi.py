import numpy as np
import torch
import torch.nn.functional as F
# loading images
from PIL import Image
from os import listdir

x_filter = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
y_filter = x_filter.transpose(0,1)

def conv2(img, filt):
    """
    filter image using pytorch
    img - input image [H, W]
    filter - Tensor [f_H, f_W]
    TODO: check the padding
    """
    H,W = img.size()
    f_H,f_W = filt.size()
    return F.conv2d(img.expand(1,1,H,W),
                    filt.expand(1,1,f_H,f_W),
                    padding=((f_H-1)//2,(f_W-1)//2)).squeeze()

def align(stack):
    """
    stack - [batch, width, height]
    output - align stack
    """

def merge(stack):
    """
    stack - align stack of images [batch, height, width]
    output - single image [height ,width]
    """


def load_images(path, N):
    """
    path - path to images directory example /tmp/exp1/ 
    N - number of images to load 
    output - stack images as pytorch tensor 
    """
    images_files = listdir(path)
    images = [path+img for img in images_files if img.endswith(".tif")]
    stack = np.asarray([np.array(Image.open(img)) for img in images], dtype=np.float32) # have to be float32 for conv2d input 
    return torch.from_numpy(stack)


if __name__ == '__main__':
    print("hello world!")
