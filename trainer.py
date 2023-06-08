from pprint import pprint
from collections import defaultdict
import argparse
import torch
import torch.backends.cudnn as cudnn

from my_dataset.contrastive_learning_dataset import ContrastiveLearningDataset
from my_dataset.augmentation import NoiseAugment
from simclr import SimCLR
from models.model import ImageEmbedding, FeatureExtractor
from data_utils import DataSourceController
from config import TrainingConfig, train_source


parser = argparse.ArgumentParser(description="PyTorch SimCLR")
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=96,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0003,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--fp16-precision",
    action="store_true",
    help="Whether or not to use 16-bit precision GPU training.",
)

parser.add_argument(
    "--out_dim", default=256, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--log-every-n-steps", default=10, type=int, help="Log every n steps"
)
parser.add_argument(
    "--temperature",
    default=0.07,
    type=float,
    help="softmax temperature (default: 0.07)",
)
parser.add_argument(
    "--n-views",
    default=2,
    type=int,
    metavar="N",
    help="Number of views for contrastive learning training.",
)
parser.add_argument("--gpu-index", default=0, type=int, help="Gpu index.")


def enable_gpu(args):
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
        args.gpu_index = -1


def get_data(train_source):
    data = DataSourceController(
        filter_=lambda x: len(x.label) < 30,
        transform=lambda x: x.replace("\u200c", "")
    )
    for k in [
        "boise_camera_train",
        "boise_scan_train",
        "boise_conjunct_train",
        "bn_htr_train",
    ]:
        data.add_data(**train_source[k])

    unique_texts = defaultdict(list)
    for _data in data.data:
        unique_texts[_data.label].append(_data)

    print(f"Total number of data: {len(data.data)}")
    print(f"Total number of Unique data: {len(unique_texts)}")

    return data.data


def main():
    args = parser.parse_args()
    pprint(args)
    enable_gpu(args)

    data = get_data(train_source)
    train_dataset = ContrastiveLearningDataset(data, noiseAugment=NoiseAugment(1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
    )

    feature_extractor = FeatureExtractor(TrainingConfig.__dict__())
    model = ImageEmbedding(FeatureExtractor=feature_extractor)

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader),
        eta_min=0,
        last_epoch=-1
    )

    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            log_dir=TrainingConfig.log_dir,
        )

        simclr.train(train_loader)


if __name__ == "__main__":
    main()
