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


train_sources = {
    "boise_camera_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/camera/camera_train_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/camera/camera_crop_images",
        "id": "boise_camera_train",
    },
    "boise_scan_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/scan/scan_train_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/scan/scan_crop_images",
        "id": "boise_scan_train",
    },
    "boise_conjunct_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/conjunct_train_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/conjunct_crop_images",
        "id": "boise_conjunct_train",
    },
    "syn_boise_conjunct_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/syn/conjunct_boise_state_train_syn/labels_train.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/syn/conjunct_boise_state_train_syn/Data",
        "id": "syn_boise_conjunct_train",
        "take_n": 5000,
    },
    "syn_train": {
        "data": "/mnt/JaHiD/Zahid/RnD/bengali_ocr/Data-Model/Synthetic-Data/20221015_16.21.35/syn300000.json",
        "base_dir": "/mnt/JaHiD/Zahid/RnD/bengali_ocr/Data-Model/Synthetic-Data/20221015_16.21.35/Data",
        "id": "syn_train",
        "take_n": 290000,
    },
    "bn_grapheme_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/bn_grapheme_dataset/bn_grapheme_train.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/bn_grapheme_dataset/images",
        "id": "bn_grapheme_train",
        # 'take_n' : 100000
    },
    "bangla_writting_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/ocr_hw_data/bangla_writting_train.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/ocr_hw_data/images_analysis_21062023/kept",
        "id": "bangla_writting_train",
        # 'take_n' : 100000
    },
    "bn_htr_train": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/bn_htrd_train.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/words",
        "id": "bn_htr_train",
        # 'take_n' : 100000
    },
}

val_sources = {
    "boise_camera_val": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/camera/camera_val_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/camera/camera_crop_images",
        "id": "boise_camera_val",
    },
    "boise_scan_val": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/scan/scan_val_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/scan/scan_crop_images",
        "id": "boise_scan_val",
    },
    "boise_conjunct_val": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/conjunct_val_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/conjunct_crop_images",
        "id": "boise_conjunct_val",
    },
    "syn_boise_conjunct_val": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/syn/conjunct_boise_state_val_syn/labels_val.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/syn/conjunct_boise_state_val_syn/Data",
        "id": "syn_boise_conjunct_val",
        # 'take_n': 3000
    },
    "syn_val": {
        "data": "/mnt/JaHiD/Zahid/RnD/bengali_ocr/Data-Model/Synthetic-Data/20221015_16.21.35/syn200000.json",
        "base_dir": "/mnt/JaHiD/Zahid/RnD/bengali_ocr/Data-Model/Synthetic-Data/20221015_16.21.35/Data",
        "id": "syn_val",
        "take_n": 100000,
    },
    "bn_grapheme_val": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/bn_grapheme_dataset/bn_grapheme_val.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/bn_grapheme_dataset/images",
        "id": "bn_grapheme_val",
        # 'take_n' : 10000
    },
    "bangla_writting_val": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/ocr_hw_data/bangla_writting_val.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/ocr_hw_data/images_analysis_21062023/kept",
        "id": "bangla_writting_val",
        # 'take_n' : 100000
    },
    "bn_htr_val": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/bn_htrd_val.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/words",
        "id": "bn_htr_val",
        # 'take_n' : 100000
    },
}

test_sources = {
    "bangla_writting_test": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/ocr_hw_data/bangla_writting_test.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/ocr_hw_data/images_analysis_21062023/kept",
        "id": "bangla_writting_test",
        # 'take_n' : 100000
    },
    "bn_htr_test": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/bn_htrd_test.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/BN-HTRd A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)/BN-HTR_Dataset/words",
        "id": "bn_htr_test",
        # 'take_n' : 100000
    },
    "boise_camera_test": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/camera/camera_test_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/camera/camera_crop_images",
        "id": "boise_camera_test",
    },
    "boise_scan_test": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/scan/scan_test_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/scan/scan_crop_images",
        "id": "boise_scan_test",
    },
    "boise_conjunct_test": {
        "data": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/conjunct_test_v2.json",
        "base_dir": "/mnt/JaHiD/Zahid/DataSet/bn_text_dataset/boiseState/conjunct/conjunct_crop_images",
        "id": "boise_conjunct_test",
    },
}
