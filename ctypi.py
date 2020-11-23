import numpy as np
import torch
import torch.nn.functional as F
# loading images
from PIL import Image
from os import listdir

y_filter = torch.Tensor([[1 / 2, 0, -1 / 2]])
x_filter = y_filter.transpose(0, 1)
y_spectral_filter = torch.Tensor([[0.0116850998497429921230139626686650444753468036651611328125,-0.0279730819380002923568717676516826031729578971862792968750,0.2239007887600356350166208585505955852568149566650390625000,0.5847743866564433234955799889576155692338943481445312500000,0.2239007887600356350166208585505955852568149566650390625000,-0.0279730819380002923568717676516826031729578971862792968750,0.0116850998497429921230139626686650444753468036651611328125]])
x_spectral_filter = y_spectral_filter.transpose(0, 1)

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

def dxdy(ref,moving,Dx=None,Dy=None,A=None):
    """
    ref - Tensor[H,W]
    moving - Tensor[H,W]
    Dx - ref x derivative Tensor[H,W]
    Dy - ref y derivative Tensor[H,W]
    """
    N = floor(len(x_filter)//2)
    if Dx==None:
        Dx = conv2(ref,x_filter)[N:-N,N:-N]
    if Dy==None:
        Dy = conv2(ref,y_filter)[N:-N,N:-N]
    if A==None:
        A = torch.Tensor([[torch.sum(Dx*Dx), torch.sum(Dx*Dy)],
                          [torch.sum(Dy*Dx), torch.sum(Dy*Dy)]])

    diff_frame_dx = conv2((moving-ref),x_spectral_filter)[N+1:-N,N+1:-N]
    diff_frame_dy = conv2((moving-ref),y_spectral_filter)[N+1:-N,N+1:-N]
    b = torch.Tensor([[torch.sum(Dx*diff_frame_dx)],
                      [torch.sum(Dy*diff_frame_dy)]])
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
    tmp = np.exp(-1.j*2*np.pi*np.fft.fftfreq(N)*dx) #TODO: remove np vector
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
    N = len(x_filter)//2
    Dx = conv2(ref,x_filter)
    Dx = conv2(Dx,x_spectral_filter)[N:-N,N:-N] #TODO: could be merged
    Dy = conv2(ref,y_filter)
    Dy = conv2(Dy,y_spectral_filter)[N:-N,N:-N] #TODO: could be merged
    A = torch.Tensor([[torch.sum(Dx*Dx), torch.sum(Dx*Dy)],
                      [torch.sum(Dy*Dx), torch.sum(Dy*Dy)]])

    for i in range(1,len(stack)):
        dx,dy = dxdy(ref,stack[i],Dx,Dy,A).numpy()
        stack[i] = shift_image(stack[i],-dx,-dy)

def merge(stack):
    """
    stack - align stack of images [batch, height, width]
    output - single image [height ,width]
    """
    return torch.mean(stack,dim=0)


def load_images(path, N):
    """
    path - path to images directory example /tmp/exp1/ 
    N - number of images to load 
    output - stack images as pytorch tensor 
    """
    images_files = listdir(path)
    images_files.sort()
    if N>0:
        images_files = images_files[:N]

    images = [path+img for img in images_files if img.endswith(".tif")]
    stack = np.asarray([np.array(Image.open(img)) for img in images], dtype=np.float32) # have to be float32 for conv2d input 
    return torch.from_numpy(stack)


if __name__ == '__main__':
    # Argument parser 
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Subpixel alignment and merging using Maor's algorithm")
    ap.add_argument('path',
                    help='dir path of the stack image in tif format')
    ap.add_argument('-N',
                    default=8,
                    help='number of images to load',
                    type=int)
    ap.add_argument('-o','--output', 
                    default='./',
                    help='output dir (default: %(default)s)')

    args = ap.parse_args()
    arr = load_images(args.path,args.N)
    align(arr)

    merged = merge(arr).numpy()
    # scale
    merged = (merged - merged.min()) / (merged.max() - merged.min())
    output = Image.fromarray((255*merged).astype(np.uint8))
    output.save(args.output+'out.tif')
    
    
