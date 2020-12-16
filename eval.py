from __future__ import print_function, division
import os

import torch

from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path", type=str, default="./saved_models/fixmatch/model_best.pth"
    )
    parser.add_argument("--use_train_model", action="store_true")

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="WideResNet")
    parser.add_argument("--net_from_name", type=bool, default=False)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    """
    Data Configurations
    """
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training. "
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = (
        checkpoint["train_model"] if args.use_train_model else checkpoint["eval_model"]
    )

    _net_builder = net_builder(
        args.net,
        args.net_from_name,
        {
            "depth": args.depth,
            "widen_factor": args.widen_factor,
            "leaky_slope": args.leaky_slope,
            "dropRate": args.dropout,
        },
    )

    _eval_dset = SSL_Dataset(
        name=args.dataset, train=False, data_dir=args.data_dir, seed=args.seed
    )

    eval_dset_basic = _eval_dset.get_dset()
    args.num_classes = _eval_dset.num_classes
    args.num_channels = _eval_dset.num_channels

    net = _net_builder(num_classes=args.num_classes, in_channels=args.channels)
    net.load_state_dict(load_model)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    eval_loader = get_data_loader(eval_dset_basic, args.batch_size, num_workers=1)

    acc = 0.0
    with torch.no_grad():
        for image, target in eval_loader:
            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)

            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()

    print(f"Test Accuracy: {acc/len(eval_dset_basic)}")
