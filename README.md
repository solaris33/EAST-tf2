
# EAST-tf2: An Efficient and Accurate Scene Text Detector in TensorFlow2

This is a TensorFlow2 & Keras implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) based on a Keras implementation made by [kurapan](https://github.com/kurapan/EAST) and a TensorFlow1 implementation made by [argman](https://github.com/argman/EAST).

The features are summarized below:

- Only RBOX geometry is implemented
- Differences from the original paper
  - Uses ResNet-50 instead of PVANet
  - Uses dice loss function instead of balanced binary cross-entropy

### Requirements

- TensorFlow 2.0 or greater
- Python 3

### Expected directory structure
```bash
├── EAST-tf2
│   ├── data
│   │   ├── ICDAR2015
│   │   │   ├── train_data
│   │   │   │   ├── gt_img_1.txt
│   │   │   │   ├── img_1.jpg
│   │   │   │   ├── ...
│   │   │   ├── test_data
│   │   │   │   ├── img_1.jpg
│   │   │   │   ├── ...
│   │   │   ├── test_data_output
│   ├── lanms
│   ├── train.py
│   ├── eval.py
│   ├── ...
```

### Data

You can use your own data, but the annotation files need to conform the ICDAR 2015 format.

ICDAR 2015 dataset can be downloaded from this [site](http://rrc.cvc.uab.es/?ch=4&com=introduction). You need the data from Task 4.1 Text Localization.

Alternatively, you can download a training dataset consisting of all training images from ICDAR 2015 and ICDAR 2013 datasets with annotation files in ICDAR 2015 format [here](https://drive.google.com/a/nlab-mpg.jp/uc?id=1p9a3K0czxIJ6zx0cFMURnKg5ydTK3jlk&export=download) (supported by [kurapan](https://github.com/kurapan/EAST)).

The original datasets are distributed by the organizers of the [Robust Reading Competition](http://rrc.cvc.uab.es/) and are licensed under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

### Training

You need to put all of your training images and their corresponding annotation files in one directory. The annotation files have to be named `gt_IMAGENAME.txt`.

Training is started by running `train.py`. It accepts several arguments including path to training data, and path where you want to save trained checkpoint models. You can see all of the arguments you can specify in the `train.py` file.

#### Execution example
```
python3 train.py --training_data_path=./data/ICDAR2015/train_data/ --checkpoint_path=./east_resnet_50_rbox
```

### Test

The images you want to detect have to be in one directory, whose path you have to pass as an argument. Detection is started by running `eval.py` with arguments specifying gpu number to run, path to the images to be detected, the trained model, and a directory which you want to save the output in.

#### Execution example
```
python3 eval.py --gpu_num=1 --test_data_path=./data/ICDAR2015/test_data --model_path=./east_resnet_50_rbox/ --output_dir=./data/ICDAR2015/test_data_output/
```

### Detection examples
![image_1](examples/img_10.jpg)
![image_2](examples/img_12.jpg)
![image_3](examples/img_13.jpg)
![image_4](examples/img_14.jpg)
![image_5](examples/img_15.jpg)
![image_6](examples/img_26.jpg)
![image_7](examples/img_28.jpg)
![image_8](examples/img_29.jpg)
![image_9](examples/img_75.jpg)

### Reference
- [kurapan's Keras EAST Implementation](https://github.com/kurapan/EAST)
- [argman's TensorFlow1 EAST Implementation](https://github.com/argman/EAST)