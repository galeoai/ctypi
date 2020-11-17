# ctypi algorithm
Maor algorithm for subpixel registration

## Description

Model: 
I2(x,y) = I1(x+dx,y+dy) \
Assuming that dx,dy << feature size \
I2(x,y) ≈ I1(x,y) + Dx(x,y)dx + Dy(x,y)dy 

## TODO 
- [X] pytorch basic implementation 
- [X] Spectral filter
- [ ] testing
- [ ] support more formats
- [ ] split the implementation and the app 
- [ ] move Tensors to the gpu if available
- [ ] Integration with Dudy's framework
- [ ] Global pixel registration 

## Getting Started

### Dependencies

* pytorch >= 1.7
* numpy >= 1.19
* PIL 

Run:
```
pip3 install -r requirements.txt
```
or
```
pip3 install numpy torch pillow
```

### Executing program

* PATH is the dir where the tif images are 
```
python3 ctypi.py PATH
```

