
class TrainingConfig():
    log_dir = "./log"
    img_width = 128
    img_height = 32
    channels = 1
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


train_config = {
    key: value for key, value in TrainingConfig.__dict__.items()
    if not key.startswith('__') and not callable(key)
}
print(train_config)