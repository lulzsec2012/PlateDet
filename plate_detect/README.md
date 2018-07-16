This proj is organized as follows:

```
plate_detect/
├── ckpt
│   ├── ckpt_416
│   └── ckpt_608
├── data
│   ├── test_list
│   ├── train_data
│   └── train_list
├── examples
├── models
├── prepare_data
└── tools
```

### Requirements
    tensorflow-gpu>=1.5.1
    python3

### Prepare data
data should be organized as follows:

```
data/
├── test_list
│   └── test.txt
├── train_data
└── train_list
    └── train.txt
```

data format in train.txt:
    image_path classes gt_box

set the target path of train.records in ${plate_detect_ROOT}/config.py:

    ```
    cd ${plate_detect_ROOT};
    vim config.py
    cfg.data_path = {your_path}
    ```

generate train.records:

    ```
    cd ${plate_detect_ROOT}/preprare_data/
    python gen_tf_records_fast.py
    ```

### Training

set the number of gpus in ${plate_detect_ROOT}/config.py:

    ```
    cfg.train.num_gpus = {num_gpus}
    ```

    ```
    cd ${plate_detect_ROOT}/examples/
    python multi_gpus_train.py
    ```

### Tesing

    ```
    cd ${plate_detect_ROOT}/examples/
    vim test.py
    g_step = {the global step of ckpt}
    image_path = {your_path}
    python test.py
    ```

### accuarcy

    ```
    cd ${plate_detect_ROOT}/examples/
    vim accuracy.py
    g_step = {the global step of ckpt}
    python accuracy.py
    ```
