# Dataset Setup Instructions

To run the unlearning pipeline, you need to set up the target datasets in this `data` folder (or another folder of your choice specified by `--data_location`).

The datasets should be structured in subdirectories matching the requirements of the code in `src/datasets/`. Below is the recommended layout:

```
data/
├── cars/                     # Stanford Cars dataset
│   ├── cars_train/
│   ├── cars_test/
│   └── cars_devkit/
│
├── dtd/                      # Describable Textures Dataset (DTD)
│   └── images/
│
├── eurosat/                  # EuroSAT (Sentinel-2 land cover)
│   └── 2750/
│
├── gtsrb/                    # German Traffic Sign Recognition Benchmark
│   ├── Final_Training/
│   └── Final_Test/
│
├── MNIST/                    # MNIST handwritten digits
│   └── raw/                  # (Auto-downloaded if missing)
│
├── resisc45/                 # NWPU-RESISC45 satellite dataset
│   └── NWPU-RESISC45/
│
├── sun397/                   # SUN397 scene classification dataset
│   └── SUN397/
│
├── svhn/                     # Street View House Numbers (SVHN)
│   # (Auto-downloaded if missing)
│
└── imagenet/                 # ImageNet (Retain Dataset)
    ├── val/
    └── train/
```

## Dataset Configuration Defaults
For details on how each dataset is initialized, preprocessed, or split into validation and test sets, refer to the loader files in the package:
- [cars.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/cars.py)
- [dtd.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/dtd.py)
- [eurosat.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/eurosat.py)
- [gtsrb.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/gtsrb.py)
- [mnist.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/mnist.py)
- [resisc45.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/resisc45.py)
- [sun397.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/sun397.py)
- [svhn.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/svhn.py)
- [imagenet.py](file:///Users/grady/Documents/research-internship/src/ModularConsensusUnlearning/src/datasets/imagenet.py)
