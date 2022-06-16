## PyNet-V2 Mobile: Efficient On-Device Photo Processing With Neural Networks

<img src="http://people.ee.ethz.ch/~ihnatova/assets/img/pynet/pynet_teaser.jpg"/>


#### 1. Overview [[Paper (in progress)]]() [[Project Webpage (in progress)]]()

This repository provides the implementation of further improvement of the PyNet model originally presented in [this paper](https://arxiv.org/abs/2002.05509). 


#### 2. Prerequisites

- Python: scipy, numpy, imageio and pillow packages
- [TensorFlow 1.X](https://www.tensorflow.org/install/) + [CUDA cuDNN](https://developer.nvidia.com/cudnn)
- Nvidia GPU


#### 3. First steps

- Download the pre-trained [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k) <sup>[Mirror](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing&resourcekey=0-Ff-0HUQsoKJxZ84trhsHpA)</sup> and put it into `vgg_pretrained/` folder.
- Download the pre-trained [PyNET model](https://drive.google.com/file/d/1txsJaCbeC-Tk53TPlvVk3IPpRw1Ro3BS/view?usp=sharing) and put it into `models/original/` folder.
- Download [Zurich RAW to RGB mapping dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset) and extract it into `raw_images/` folder.    
  <sub>This folder should contain three subfolders: `train/`, `test/` and `full_resolution/`</sub>
  
  <sub>*Please note that Google Drive has a quota limiting the number of downloads per day. To avoid it, you can login to your Google account and press "Add to My Drive" button instead of a direct download. Please check [this issue](https://github.com/aiff22/PyNET/issues/4) for more information.* </sub>



#### 4. PyNet-V2 Mobile CNN

[WIP]


#### 5. Training the model

The model is trained level by level, starting from the lowest. The script below incorporates all training steps:

```bash
./train.sh
```


#### 6. Test the provided pre-trained models on full-resolution RAW image files

```bash
python test_model_keras.py
```

Optional parameters:

>```--model```: - &nbsp; path to the Keras model checkpoint <br/>
>```--inp_path```: **```raw_images/test/```** &nbsp; - &nbsp; path to the folder with **Zurich RAW to RGB dataset** <br/>
>```--out_path```: **```.```** &nbsp; - &nbsp; path to the output images <br/>


#### 7. Folder structure

>```models/```            &nbsp; - &nbsp; logs and models that are saved during the training process <br/>
>```models/original/```   &nbsp; - &nbsp; the folder with the provided pre-trained PyNET model <br/>
>```raw_images/```        &nbsp; - &nbsp; the folder with Zurich RAW to RGB dataset <br/>
>```results/```           &nbsp; - &nbsp; visual results for small image patches that are saved while training <br/>
>```results/full-resolution/``` &nbsp; - &nbsp; visual results for full-resolution RAW image data saved during the testing <br/>
>```vgg-pretrained/```    &nbsp; - &nbsp; the folder with the pre-trained VGG-19 network <br/>

>```load_dataset.py```    &nbsp; - &nbsp; python script that loads training data <br/>
>```model.py```           &nbsp; - &nbsp; PyNET implementation (Keras) <br/>
>```train_model_keras.py```     &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model_keras.py```      &nbsp; - &nbsp; applying the pre-trained model to full-resolution test images <br/>
>```utils.py```           &nbsp; - &nbsp; auxiliary functions <br/>
>```vgg.py```             &nbsp; - &nbsp; loading the pre-trained vgg-19 network <br/>


#### 9. Bonus files

These files can be useful for further experiments with the model / dataset:

>```dng_to_png.py```            &nbsp; - &nbsp; convert raw DNG camera files to PyNET's input format <br/>
>```ckpt2pb_keras.py```     &nbsp; - &nbsp; converts Keras checkpoint to TFLite format <br/>
>```evaluate_accuracy_tflite.py```     &nbsp; - &nbsp; compute PSNR and MS-SSIM scores on Zurich RAW-to-RGB dataset for TFLite model <br/>


#### 10. License

Copyright (C) 2022 Andrey Ignatov. All rights reserved.

Licensed under the [CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

The code is released for academic research use only.


#### 11. Citation
[WIP] 


#### 12. Any further questions?

```
Please contact Andrey Ignatov (andrey@vision.ee.ethz.ch) for more information
```
