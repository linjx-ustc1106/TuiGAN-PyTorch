# TuiGAN-PyTorch
Official PyTorch Implementation of "TuiGAN: Learning Versatile Image-to-Image Translation with Two Unpaired Images"

## TuiGAN's applications
TuiGAN can be use for various computer vision tasks ranging from image style transfer to object transformation and appearance transformation:
 ![](imgs/examples.jpg)

## Usage

### Install dependencies

```
python -m pip install -r requirements.txt
```

Our code was tested with python 3.6  and PyToch 1.0.0

###  Train
To train TuiGAN model on two unpaired images, put the first training image under `datas/task_name/trainA` and the second training image under `datas/task_name/trainB`, and run

```
python train.py --input_name <task_name>
```

##  Comparison Results

###  General Unsupervised Image-to-Image Translation
![](imgs/comparisons.jpg)
###  Image Sytle Transfer
![](imgs/style.jpg)
###  Animal Face Translation
![](imgs/dog.jpg)
###  Painting-to-Image Translation
![](imgs/trees.jpg)
