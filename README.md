# Fast Perceptual Image Enhancement [[Paper]](https://arxiv.org/abs/1812.11852)

#### Prerequisites

- Python + scipy, numpy packages
- [TensorFlow (>=1.0.1)](https://www.tensorflow.org/install/) + [CUDA CuDNN](https://developer.nvidia.com/cudnn)
- Nvidia GPU


#### First steps

- Download the pre-trained [VGG-19 model](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing) and put it into `vgg_pretrained/` folder
- Download [DPED dataset](http://people.ee.ethz.ch/~ihnatova/#dataset) (patches for CNN training) and extract it into `dped/` folder.  
<sub>This folder should contain three subolders: `sony/`, `iphone/` and `blackberry/`, but only `iphone/` is needed.</sub>


#### Train the model

```bash
python train_model.py model=iphone num_train_iters=40000 run=replication convdeconv depth=16
```


#### Test the provided pre-trained model

```bash
python test_model.py model=iphone_orig test_subset=fullresolution=orig use_gpu=true
```

Optional parameters:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; the resolution of the test images [**```orig```** means original resolution]<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>


#### Test the obtained model

```bash
python test_model.py model=iphone iteration=[40000] test_subset=full resolution=orig use_gpu=true run=replication convdeconv depth=16
```

Optional parameters:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```iteration```: **```all```** or **```<number>```**  &nbsp; - &nbsp; get visual results for all iterations or for the specific iteration,  
>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**```<number>```** must be a multiple of ```eval_step``` <br/>
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; the resolution of the test 
images [**```orig```** means original resolution]<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>


#### Folder structure

>```dped/```              &nbsp; - &nbsp; the folder with the DPED dataset <br/>
>```models/```            &nbsp; - &nbsp; logs and models that are saved during the training process <br/>
>```models_orig/```       &nbsp; - &nbsp; the provided pre-trained models for **```iphone```**, **```sony```** and **```blackberry```** <br/>
>```results/```           &nbsp; - &nbsp; visual results for small image patches that are saved while training <br/>
>```summaries/```         &nbsp; - &nbsp; TensorBoard summaries generated while training <br/>
>```vgg-pretrained/```    &nbsp; - &nbsp; the folder with the pre-trained VGG-19 network <br/>
>```visual_results/```    &nbsp; - &nbsp; processed [enhanced] test images <br/>

>```load_dataset.py```    &nbsp; - &nbsp; python script that loads training data <br/>
>```models.py```          &nbsp; - &nbsp; architecture of the image enhancement [resnet] and adversarial networks <br/>
>```ssim.py```            &nbsp; - &nbsp; implementation of the ssim score <br/>
>```train_model.py```     &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model.py```      &nbsp; - &nbsp; applying the pre-trained models to test images <br/>
>```utils.py```           &nbsp; - &nbsp; auxiliary functions <br/>
>```vgg.py```             &nbsp; - &nbsp; loading the pre-trained vgg-19 network <br/>
