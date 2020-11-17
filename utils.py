import os, glob
import time
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
import logging


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(
                f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}"
            )
        setattr(cls, key, kwargs[key])


def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = "hello"

    test_cls = _test_cls()
    config = {"a": 3, "b": "change_hello", "c": 5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")


def net_builder(net_name, from_name: bool, net_conf=None):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models

        model_name_list = sorted(
            name
            for name in models.__dict__
            if name.islower()
            and not name.startswith("__")
            and callable(models.__dict__[name])
        )

        if net_name not in model_name_list:
            assert Exception(
                f"[!] Networks' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}"
            )
        else:
            return models.__dict__[net_name]

    else:
        if net_name == "WideResNet":
            import models.nets.wrn as net

            builder = getattr(net, "build_WideResNet")()
            setattr_cls_from_kwargs(builder, net_conf)
            return builder.build
        elif net_name == "efficientNet":
            return lambda num_classes: EfficientNet.from_name(
                "efficientnet-b0", num_classes=num_classes
            )
        elif "efficientnet" in net_name:
            return lambda num_classes: EfficientNet.from_name(
                net_name, num_classes=num_classes
            )
        else:
            assert Exception("Not Implemented Error")


def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)


def get_logger(name, save_path=None, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, "log.txt"))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dir_str(args):
    dir_name = (
        args.dataset
        + "/FixMatch_arch"
        + args.net
        + "_batch"
        + str(args.batch_size)
        + "_confidence"
        + str(args.p_cutoff)
        + "_lr"
        + str(args.lr)
        + "_nclass"
        + str(args.num_classes)
        + "_uratio"
        + str(args.uratio)
        + "_wd"
        + str(args.weight_decay)
        + "_wu"
        + str(args.ulb_loss_ratio)
        + "_seed"
        + str(args.seed)
        + "_numlabels"
        + str(args.num_labels)
        + "_opt"
        + str(args.opt)
    )
    return dir_name


def get_model_checkpoints(folderpath):
    """Returns all the latest checkpoint files and used parameters in the below folders

    Args:
        folderpath (str): path to search (note only depth 1 below will be searched.)

    Returns:
        list,list: lists of checkpoint names and associated parameters
    """
    # Find present models
    folderpath = folderpath.replace("\\", "/")
    model_files = glob.glob(folderpath + "/**/*.pth", recursive=True)
    folders = [model_file.split("model_best.pth")[0] for model_file in model_files]

    checkpoints = []
    params = []
    for file, folder in zip(model_files, folders):
        checkpoints.append(file)
        params.append(decode_parameters_from_path(folder))

    return checkpoints, params


def decode_parameters_from_path(filepath):
    """Decodes the parameters encoded in the filepath to a checkpoint

    Args:
        filepath (str): full path to checkpoint folder

    Returns:
        dict: dictionary with all parameters
    """
    params = {}

    filepath = filepath.replace("\\", "/")
    filepath = filepath.split("/")

    param_string = filepath[-2]
    param_string = param_string.split("_")

    params["dataset"] = filepath[-3]
    params["net"] = param_string[1][4:]
    params["batch"] = int(param_string[2][5:])
    params["confidence"] = float(param_string[3][10:])
    # params["filters"] = int(param_string[4][7:])
    params["lr"] = float(param_string[4][2:])
    params["num_classes"] = int(param_string[5][6:])
    params["uratio"] = int(param_string[6][6:])
    params["wd"] = float(param_string[7][2:])
    params["wu"] = float(param_string[8][2:])
    params["seed"] = float(param_string[9][4:])
    params["numlabels"] = int(param_string[10][9:])
    params["opt"] = param_string[11][3:]
    return params

