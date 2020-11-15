import numpy as np
import torch 
from PIL import Image # loading images
from os import listdir


def align(stack):
    """
    stack - [batch, width, height]
    output - align stack
    """


def merge(stack):
    """
    stack - align stack of images [batch, width, height]
    output - single image [width, height]
    """


def load_images(path, N):
    """
    path - path to images directory example /tmp/exp1/ 
    N - number of images to load 
    output - stack images as pytorch tensor 
    """
    images_files = listdir(path)
    images = [path+img for img in images_files if img.endswith(".tif")]
    stack = np.asarray([np.array(Image.open(img)) for img in images], dtype=np.int32)
    return torch.from_numpy(stack)
    
    
        
    

if __name__ == '__main__':
    print("hello world!")
    

