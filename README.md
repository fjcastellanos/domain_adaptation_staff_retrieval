# Domain Adaptation for Staff-Region Retrieval of Music Score Images

This repository corresponds to the paper: **Domain Adaptation for Staff-Region Retrieval of Music Score Images**, submitted to the journal 'International Journal on Document Analysis and Recognition'. 

## BibTex Reference
Not available yet.

## Using the program

To use the program, there are a series of parameters to be configured. There are two manners of using the code: by python command through console or by the script.sh included in this repository.

### Python command

By using the main.py file. Next we define the different parameters available. Note that data used for this code was extracted from [MuReT tool](https://muret.dlsi.ua.es/muret/#/). Note that the corpora contain json files with the ground-truth information and the images. 

#### Parameters

  **-type** : Execution mode: **cnn** for supervised model (Selectional Auto-encoder) and **dann** for our proposal. \
  **-gpu** : GPU index. It is possible to change the gpu used for the process if more GPUs are available. Default: 0 \
  **-db1_name** : name of the source domain. This is used for logs. \
  **-db2_name** : name of the target domain. This is used for logs. \
  **-db1_train** : path to a txt file with the list of paths to the json files to be used for training. For the **source** domain.\
  **-db1_val** : path to a txt file with the list of paths to the json files to be used for validation. For the **source** domain.\
  **-db1_test** : path to a txt file with the list of paths to the json files to be used for testing. For the **source** domain.\
  **-db2_train** : path to a txt file with the list of paths to the json files to be used for training. For the **target** domain.\
  **-db2_val** : path to a txt file with the list of paths to the json files to be used for validation. For the **target** domain.\
  **-db2_test** : path to a txt file with the list of paths to the json files to be used for testing. For the **target** domain.\
  **-classes** : list of names of the region types considered for the experimentation. This code uses the data configuration provided by the [MuReT tool] Default: **"staff" "empty-staff"**. \
  **-w** : size of the resized image. Default: 512 px (squared). \
  **-l** : Number of convolutional blocks for the encoder and the decoder. Default: 3 \
  **-f** : Number of filters. Default: 32 \
  **-k** : Kernel size. Default: 3 \
  **-l** : To configure a dropout ratio. Default: 0 \
  **-lda** : Initial value for lambda in the DANN type. Default: 0.01 \
  **-lda_inc** : Value to increment lambda each epoch. Default: 0.001 \
  **-e** : Maximum number of epochs. Default: 300 \
  **-b** : Batch size. Default: 12 \
  **-se** : Number of superepochs. Default: 1 \
  **-gpos** : Position of the GRL connection. Value 0 is the latent code (center of the SAE), negative positions are in the encoder and positive positions in the decoder.. Default: 2 \
  **-opt** : Optimizer. Default: sgd \
  **-pre** : Number of epochs to pretrain the model with only the source domain. Default: 50 \
  **-th_iou** : In test mode, the IoU minimum to consider the prediction as True Positive. Default: 0.5 \
  **-th** : Threshold used for determine the belonging of each pixel to a region accordint to the probability provided by the model. Default: 0.5 \
  **--test**: Optional parameter. When it is used, the code used the models to evaluate the test partitions. \


#### Example of use

```[python]
  python -u main.py 
                        -type dann 
                        -gpu 0 
                        
                        -db1_name Guatemala 
                        -db1_train datasets/JSON/Folds/Guatemala/fold0/train.txt 
                        -db1_val datasets/JSON/Folds/Guatemala/fold0/val.txt 
                        -db1_test datasets/JSON/Folds/Guatemala/fold0/test.txt 
                        
                        -db2_name b-59-850 
                        -db2_train datasets/JSON/Folds/b-59-850/fold0/train.txt 
                        -db2_val datasets/JSON/Folds/b-59-850/fold0/val.txt 
                        -db2_test datasets/JSON/Folds/b-59-850/fold0/test.txt 
                        
                        -classes staff empty-staff 
                        
                        -w 512 
                        -l 3 -f 32 -k 3 -drop 0 
                        -lda 0.01 
                        -lda_inc 0.001 
                        -e 300 -se 1 -b 12 
                        -gpos 2 
                        -opt sgd 
                        -pre 50 
                        -th_iou 0.5 
                        -th 0.5 
```
                        
### Script

The file script.sh contains the call to the main.py code with the parameters mentioned before. It is possible to modify the parameters to try with different variations according to the requirements.

#### Example of use
First modifying the parameters if it is necessary within script.sh and then running the script:

```
  ./script.sh
```

## Considerations

The program is prepared to keep the same name for images and ground-truth json files for matching them. Corpora tested with this program was extracted from [MuReT tool](https://muret.dlsi.ua.es/muret/#/), including the json files and the images.

Note that the data is given by txt files with lists of the path files to be used in the experiments. Therefore, it is necessary to prepare these files to be able to use the program. 

In our case, the datasets were organized as follows:

```
datasets 
  ├── JSON 
  │   ├── Folds 
  |   |   ├── b-59-850
  |   |   |   ├── fold0 
  |   |   |   |   ├── train.txt 
  |   |   |   |   ├── val.txt 
  |   |   |   |   └── test.txt 
  |   |   |   └── fold1 
  |   |   |       ├── train.txt 
  |   |   |       ├── val.txt 
  |   |   |       └── test.txt 
  |   |   └── Patriarca
  |   |       ├── fold0 
  |   |       |   ├── train.txt 
  |   |       |   ├── val.txt 
  |   |       |   └── test.txt 
  |   |       └── fold1 
  |   |           ├── train.txt 
  |   |           ├── val.txt 
  |   |           └── test.txt 
  |   ├── b-59-850 
  |   |   ├── image1.jpg.json 
  |   |   └── image2.jpg.json
  |   ├── Patriarca 
  |   |   ├── image1.jpg.json 
  |   |   └── image2.jpg.json 
  └── SRC 
      ├── b-59-850 
      |   ├── image1.jpg 
      |   └── image2.jpg 
      └── Patriarca 
          ├── image1.jpg 
          └── image2.jpg 
```
