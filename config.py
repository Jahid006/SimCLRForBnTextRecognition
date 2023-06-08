class TrainingConfig:
    log_dir = './log'
    img_width = 128
    img_height = 32
    d_model = 512
    n_heads = 8
    n_layers = 2
    dropout = 0
    leaky_relu = False

    in_channel = 1
    channels = [in_channel, 32, 64, 128, 128, 256, 256, 512, 512]
    kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 2]
    strides = [1, 1, 1, 1, 1, 1, 1, 1]
    paddings = [1, 1, 1, 1, 1, 1, 1, 0]
    batch_normms = [1, 1, 1, 1, 1, 1, 1, 0]
    max_pooling = [(2, 1), (2, 1), (0, 0), (2, 1), (0, 0), (0, 0), (2, 1), (0, 0)]


mapper = {
    "boise_camera_train": "real",
    "boise_scan_train": "real",
    "boise_conjunct_train": "real",
    "syn_train": "syn",
    "boise_camera_val": "real",
    "boise_scan_val": "real",
    "boise_conjunct_val": "real",
    "syn_val": "syn",
    "syn_boise_conjunct_train": "syn",
    "bn_grapheme_train": "bn_grapheme",
    "syn_boise_conjunct_val": "real",
    "bn_grapheme_val": "bn_grapheme",
}

train_source = {
    "boise_camera_train": {
        "data": "/home/jahid/Music/bn_dataset/boiseState/camera/split/train_annotaion.json",
        "base_dir": "/home/jahid/Music/bn_dataset/boiseState/camera/split/train_crop_images",
        "id": "boise_camera_train",
    },
    "boise_scan_train": {
        "data": "/home/jahid/Music/bn_dataset/boiseState/scan/split/train_annotaion.json",
        "base_dir": "/home/jahid/Music/bn_dataset/boiseState/scan/split/train_crop_images",
        "id": "boise_scan_train",
    },
    "boise_conjunct_train": {
        "data": "/home/jahid/Music/bn_dataset/boiseState/conjunct/split/train_annotaion.json",
        "base_dir": "/home/jahid/Music/bn_dataset/boiseState/conjunct/split/train_crop_images",
        "id": "boise_conjunct_train",
    },
    "syn_boise_conjunct_train": {
        "data": "/home/jahid/Music/bn_dataset/boiseState/conjunct/syn/conjunct_boise_state_train_syn/labels_train.json",
        "base_dir": "/home/jahid/Music/bn_dataset/boiseState/conjunct/syn/conjunct_boise_state_train_syn/Data",
        "id": "syn_boise_conjunct_train",
        "take_n": 5000,
    },
    "bn_htr_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/bn_htrd_train.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/words",
        "id": "bn_htr_train",
        # 'take_n' : 100000
    },
}

