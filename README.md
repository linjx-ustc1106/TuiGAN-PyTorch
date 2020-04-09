# TuiGAN-PyTorch
Official PyTorch Implementation of "TuiGAN: Learning Versatile Image-to-Image Translation with Two Unpaired Images"

## TuiGAN's applications
TuiGAN can be use for various computer vision tasks ranging from image style transfer to object transformation and appearance transformation:
 ![](imgs/examples.jpg)

###  Train
To train TuiGAN model on two unpaired images, put the first training image under data/task_name/trainA and the second training image under data/task_name/trainB, and run

```
python train.py --input_name <task_name>
```

###  Comparison Results

####  General Unsupervised Image-to-Image Translation

####  Image Sytle Transfer

####  Animal Face Translation

####  Painting-to-Image Translation
