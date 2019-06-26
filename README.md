# 3d-SMRnet:  MPI system matrix recovery
This repository contains the official code and model weights for the following arXiv [paper](https://arxiv.org/abs/1905.03026) (Submitted to MICCAI 2019):

```
@ARTICLE{Baltruschat3dSMRNet2019,
    author="Baltruschat, Ivo Matteo and Szwargulski, Patryk and Griese, Florian and Grosser, Mirco and Werner, Ren{\'e} and Knopp, Tobias",
    title="3d-SMRnet: Achieving a new quality of MPI system matrix recovery by deep learning",
    journal="arXiv",
    year="2019",
}
```

## Requirements
The code has been tested with Python 3.6 on Ubuntu 16.04 LTS and Windows 10. Use the following command to install all required Python packages:
```
pip install -r requirements.txt
```


## Usage
We provide the system matrices: [Perimag](https://drive.google.com/open?id=1LC7Pn0z65JWHr0IYwqMiAP_ZLpU6JrcS) and [Synomag-D](https://drive.google.com/open?id=1MGu4_YACg-vo-s1E7HHrJgLrimec0Vjh)  
Both matrices are already processed and contain the RGB-encoded ```key="Data"``` and the complex ```key="DataImag"; key="DataReal"``` data. We use a threshold of SNR=3 for the frequencies. 
For downloading our trained model weights, please see [here](https://github.com/Ivo-B/3dSMRnet/releases/latest).

### Prepare data
We only provide raw system matrices. Hence, we need to prepare the data first. The script "pre_processing.py" will create several HDF5-files with training, validation, and testing splits. 
Furthermore, LR system matrices with equidistant subsampling of the HR system matrix are created.
1. Modify the script file ```scripts/pre_processing.py```  
Update all path specific parameters
2. Run command: ```python pre_processing.py```

### How to Test
Test SR-RRDB model with 2 channels (Image/Real) and a up-scaling of 4
1. Modify the configuration file ```experiments/001_Test_SR-RRDB-3d_complex_scale4.json```  
Update all path specific parameters
2. Run command: ```python test.py -opt experiments/001_Test_SR-RRDB-3d_complex_scale4.json```

Test SR-RRDB model with 3 channels (RGB) and a up-scaling of 4
1. Modify the configuration file ```experiments/002_Test_SR-RRDB-3d_RGB_scale4.json```  
Update all path specific parameters
2. Run command: ```python test.py -opt experiments/002_Test_SR-RRDB-3d_RGB_scale4.json```

### How to Train
Train SR-RRDB model with 2 channels (Image/Real) and a up-scaling of 4
1. Modify the configuration file ```experiments/001_Test_SR-RRDB-3d_complex_scale4.json```
2. Run the command: ```python train.py -opt experiments/001_Test_SR-RRDB-3d_complex_scale4.json```

Train SR-RRDB model with 3 channels (RGB) and a up-scaling of 4
1. Modify the configuration file ```experiments/002_Test_SR-RRDB-3d_RGB_scale4.json```
2. Run the command: ```python train.py -opt experiments/002_Test_SR-RRDB-3d_RGB_scale4.json```

## Our trained models


## Results
    
    
## Acknowledgement
- Code architecture is an extension of [BasicSR](https://github.com/xinntao/BasicSR) and [LapSRN](https://github.com/twtygqyy/pytorch-LapSRN)
