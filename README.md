## 1. Installation

Reference environment:

- Python 3.11
- PyTorch 2.6.0
- torchvision 0.21.0
- timm 1.0.28

Create and activate a Conda environment:

```bash
conda create -n jumcnet python=3.11 -y
conda activate jumcnet
```

Install the dependencies from the project root directory. The following commands use the CUDA 12.4 build of PyTorch:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt timm==1.0.28
```

For other hardware configurations, select a compatible build from the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

## 2. Data Preparation

**Jujube12000**

Public dataset page: [Mendeley Data](https://data.mendeley.com/datasets/5hgxvtfvn5/1).

Organize the images as follows, preserving the filenames listed in the provided split files:

```text
dataSets/Jujube12000/
├── bianxing/    # Deformed
├── gantiao/     # Shriveled
├── haozao/      # Normal
├── liekou/      # Cracked
├── meibian/     # Moldy
└── niaozhuo/    # Bird-pecked
```

**Cassava**

Download page: [Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data).

Download the labeled training images and extract them into:

```text
dataSets/Cassava/
└── train_images/
    ├── <image_id>.jpg
    └── ...
```

The class order is CBB, CBSD, CGM, CMD, and Healthy, corresponding to labels 0–4, respectively.

**Dataset Splits**

Use the fixed split files provided in the project:

```text
splits/
├── Jujube12000/
│   └── split_seed_2222/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
└── Cassava/
    └── split_seed_1111/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

| Dataset | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Jujube12000 | 8,115 | 1,155 | 2,325 | 11,595 |
| Cassava | 14,975 | 2,136 | 4,286 | 21,397 |

Each line in a split file contains the relative image path, class index, and class name, separated by tabs. There is no need to regenerate the provided splits.

Use of each dataset is subject to the terms specified by its provider.
