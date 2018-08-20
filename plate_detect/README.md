This proj is organized as follows:

```
plate_detect/
├── ckpt
│   └── ckpt_608
├── data
│   ├── test_data
│   └── train_data
├── examples
├── models
├── prepare_data
└── tools
```

### Requirements
    tensorflow-gpu>=1.5.1
    python3

### Main Results
    ######the accuracy of current version: 94.56% 
    the current version: accuracy = 98.33%, recall = 99.31%

### Prepare data
data should be organized as follows:

```
data/
├── test_data
│   ├── 9999388_20180205114702178_京ADA098_sp.jpg
│   └── plate_detect_test.records
└── train_data
    └── plate_detect_train.records
```

### Training

1. Download dataset:
    cd ${plate_detect_ROOT}/data/train_data/
    wget http://193.169.1.235/plate_detect_train.records

2. set the number of gpus in ${plate_detect_ROOT}/config.py:

    ```
    cfg.train.num_gpus = {num_gpus}
    ```

    ```
    cd ${plate_detect_ROOT}/examples/
    python multi_gpus_train.py
    ```

### Tesing

1. Download checkpoint model:
    cd ${plate_detect_ROOT}/ckpt/
    wget http://193.169.1.235/plate_detect_original_model.zip
    unzip plate_detect_original_model.zip
    rm plate_detect_original_model.zip

2.
    ```
    default:
        cd ${plate_detect_ROOT}/examples/
        python test.py
    if you want to use your images and ckpt model to test:
        cd ${plate_detect_ROOT}/examples/
        vim test.py
        g_step = {the global step of ckpt}
        image_path = {your_path}
        python test.py
    ```

### accuarcy

1. Download test data:
    cd ${plate_detect_ROOT}/data/test_data/
    wget http://193.169.1.235/plate_detect_test.records

2.
    ```
    cd ${plate_detect_ROOT}/examples/
    vim accuracy.py
    g_step = {the global step of ckpt}
    python accuracy.py
    ```
