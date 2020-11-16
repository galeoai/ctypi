import numpy as np
import torch
import torch.nn.functional as F
# loading images
from PIL import Image
from os import listdir

x_filter = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
y_filter = x_filter.transpose(0,1)

def complex_mul(A,B):
    """
    out = A*B for complex torch.tensors
    A - [a, b, 2] if vector a or b should be 1
    B - [b, c, 2]
    out - [a, c, 2]
    """
    A_real, A_imag = A[...,0], A[...,1]
    B_real, B_imag = B[...,0], B[...,1]
    return torch.stack([A_real@B_real-A_imag@B_imag,
                        A_real@B_imag+A_imag@B_real],
                       dim=-1)

def complex_vector_brodcast(A,B):
    """
    out = A*B for complex torch.tensors
    A - [1, b, 2]
    B - [a, b, 2]
    out - [a, b, 2]
    """
    A_real, A_imag = A[...,0], A[...,1]
    B_real, B_imag = B[...,0], B[...,1]
    return torch.stack([A_real*B_real-A_imag*B_imag,
                        A_real*B_imag+A_imag*B_real],
                       dim=-1)


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

def dxdy(ref,moving,Dx,Dy,A=None):
    """
    ref - Tensor[H,W]
    moving - Tensor[H,W]
    Dx - ref x derivative Tensor[H,W]
    Dy - ref y derivative Tensor[H,W]
    """
    if A==None:
        A = torch.Tensor([[torch.sum(Dx*Dx), torch.sum(Dx*Dy)],
                          [torch.sum(Dy*Dx), torch.sum(Dy*Dy)]])

    b = torch.Tensor([[torch.sum(Dx*(moving-ref))],
                      [torch.sum(Dy*(moving-ref))]])
    return torch.solve(b, A)[0]  # return the result only

def shift_image(img, dx, dy):
    """
    img - Tensor[H,W]
    dx - float
    dy - float
    """
    N,M = img.size()
    #fft needs the last dim to be 2 (real,complex) TODO: faster implementation
    img_padded = torch.stack((img,torch.zeros(N,M)),dim=2)
    fft_img = torch.fft(img_padded,2)
    tmp = np.exp(-1.j*2*np.pi*np.fft.fftfreq(N)*dx)
    X = torch.from_numpy(tmp.view("(2,)float")).float()
    tmp = np.exp(-1.j*2*np.pi*np.fft.fftfreq(M)*dy)
    Y = torch.from_numpy(tmp.view("(2,)float")).float()
    # clac the shifted image
    tmp = complex_vector_brodcast(fft_img,X.unsqueeze(1))
    tmp = complex_vector_brodcast(Y.unsqueeze(0),tmp)
    return torch.ifft(tmp,2).norm(dim=2)


def align(stack):
    """
    stack - [batch, width, height]
    output - align stack
    """
    ref = stack[0] # set the first frame as refernce
    # clac derivative and A matrix
    Dx = conv2(ref,x_filter)
    Dy = conv2(ref,y_filter)
    A = torch.Tensor([[torch.sum(Dx*Dx), torch.sum(Dx*Dy)],
                      [torch.sum(Dy*Dx), torch.sum(Dy*Dy)]])

    for img in stack[1:]:
        xy = dxdy(ref,img,Dx,Dy,A)

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
    arr = load_images("/home/dicker/workspace/ctypi/images/",8)
    import matplotlib.pyplot as plt
