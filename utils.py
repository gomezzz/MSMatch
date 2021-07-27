import os, glob
import time
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from random import sample
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def plot_cmatrix(preds,labels,encoding, figsize=(8, 5),dpi=150, class_names_font_scale=1.2, matrix_font_size=12, save_fig_name=None):
    """Plotting the confusion matrix for one or three dataset seeds. 

    Args:
        preds ([numpy array]): array containing predictions for one or three dataset seeds.
        labels ([numpy array]):  array containing labels for one or three dataset seeds.
        encoding ([list]): classes label encoding.
        figsize (tuple, optional): size of the output figure. Defaults to (8, 5).
        dpi (int, optional): Dots for inch. Defaults to 150.
        class_names_font_scale (float, optional): Font scale for class names in the confusion matrix. Defaults to 1.2.
        matrix_font_size (int, optional): font size of the confusion matrix numbers. Defaults to 12.
        save_fig_name ([str], optional): output figure name. If 'None', no output figure is saved. Defaults to None.
    """
    if len(preds) > 1:
        n = 0
        for preds_seed, labels_seed in zip(preds, labels):
            if n == 0:
                cm = confusion_matrix(labels_seed,preds_seed,normalize='true')
                n+=1
            else:
                cm+= confusion_matrix(labels_seed,preds_seed,normalize='true')
        cm/=len(preds)
    else:
        cm = confusion_matrix(labels,preds,normalize='true')
        
    cm=np.floor(cm*1000)/10
    sn.set(font_scale=class_names_font_scale) # for label size
    plt.figure(figsize=figsize, dpi=dpi)
    df_cm=pd.DataFrame(cm, index=[k for k in encoding], columns=[k for k in encoding])
    labels=df_cm.applymap(lambda v: str(int(round(v))) if int(round(v)) > 0 else '')
    sn.heatmap(df_cm, annot=labels, linewidths=.5, fmt='', annot_kws={'fontsize':matrix_font_size}) # font size
    if save_fig_name is not None:
        sn.set_theme()
        plt.tight_layout()
        plt.savefig(save_fig_name)


def plot_examples(images,labels,encoding, figsize=(8, 5),dpi=150, labels_fontsize=5, prediction=None, save_fig_name=None):
    """Plotting 32 randomly sampled image examples for a target dataset, ensuring that at least one image for each class is got. If `prediction` is given, both predicted and expected classes are shown for each image.

    Args:
        images ([list]): list of images to plot.
        labels ([list]): list of predicted classes.
        encoding ([list]): classes label encoding.
        figsize (tuple, optional): size of the output figure. Defaults to (8, 5).
        dpi (int, optional): Dots for inch. Defaults to 150.
        labels_fontsize ([str]): label fontsize. Default to 5.
        prediction ([list], optional): List of predicted classes. Defaults to None.
        save_fig_name ([str], optional): output figure name. If 'None', no output figure is saved. Defaults to None.
    """
    def sort_x_according_to_y(x,y):
        return [x for _,x in sorted(zip(y,x))]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    class_found=[]
    shuffled_idx=list(np.random.permutation(len(labels)))
    
    labels=sort_x_according_to_y(labels, shuffled_idx)
    images=sort_x_according_to_y(images, shuffled_idx)
    if prediction is not None:
        prediction=sort_x_according_to_y(prediction, shuffled_idx)
        
        
    labels_idx=[]
    class_found=[]

    for l in range(len(labels)):
        if not(labels[l] in class_found):
            labels_idx.append(l)
            class_found.append(labels[l])
            
    print("Number of different classes found:", len(class_found))
            
    n_to_add= 32 - len(labels_idx)
            
    for l in range(len(labels)):
        if n_to_add == 0:
            break
            
        if not(l in labels_idx):
            labels_idx.append(l)
            n_to_add-=1
            
    
    #rand_indices=sample(range(len(images)), 32)
    for idx, rand_idx in enumerate(labels_idx):
        img = images[rand_idx]
        ax = fig.add_subplot(4, 8, idx+1, xticks=[], yticks=[])
        if np.max(img) > 1.5:
            img = img / 255
        plt.imshow(img)
        if prediction is not None:
            label = "GT: " + encoding[labels[rand_idx]] + "\n PR: " + encoding[prediction[rand_idx]]
        else:
            label = encoding[labels[rand_idx]]    
        plt.title(str(label),fontsize=labels_fontsize)

    if save_fig_name is not None:
        plt.savefig(save_fig_name)


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


def net_builder(
    net_name, from_name: bool, net_conf=None, pretrained=False, in_channels=3
):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
        pre_trained: Specifies if a pretrained network should be loaded (only works for efficientNet)
        in_channels: Input channels to the network
    """
    if from_name:
        assert in_channels == 3
        assert not pretrained
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
            assert in_channels == 3
            assert not pretrained
            import models.nets.wrn as net

            builder = getattr(net, "build_WideResNet")()
            setattr_cls_from_kwargs(builder, net_conf)
            return builder.build
        elif "efficientnet" in net_name:
            if pretrained:
                print("Using pretrained", net_name, "...")
                return lambda num_classes, in_channels: EfficientNet.from_pretrained(
                    net_name, num_classes=num_classes, in_channels=in_channels
                )

            else:
                print("Using not pretrained model", net_name, "...")
                return lambda num_classes, in_channels: EfficientNet.from_name(
                    net_name, num_classes=num_classes, in_channels=in_channels
                )
        else:
            assert Exception("Not Implemented Error")


def test_net_builder(net_name, from_name, net_conf=None, pretrained=False):
    builder = net_builder(net_name, from_name, net_conf, pretrained)
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
    if args.pretrained:
        dir_name = dir_name + "_pretrained"
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
    model_files = glob.glob(folderpath + "/**/model_best.pth", recursive=True)
    folders = [model_file.split("model_best.pth")[0] for model_file in model_files]

    checkpoints = []
    params = []
    for file, folder in zip(model_files, folders):
        checkpoints.append(file)
        params.append(decode_parameters_from_path(folder))

    return checkpoints, params


def _read_best_iteration_number(folder):
    """Reads from the run log file at which iteration the best result was obtained.

    Args:
        folder (str): results folder

    Returns:
        int: iteration number
    """
    # Read second last line from the file
    with open(folder + "log.txt", "r") as file:
        lines = file.read().splitlines()
        second_last_line = lines[-2]

    # Fine iteration number
    iteration_str = second_last_line.split(", at ")[1]
    return int(iteration_str.split(" iters")[0])


def decode_parameters_from_path(filepath):
    """Decodes the parameters encoded in the filepath to a checkpoint

    Args:
        filepath (str): full path to checkpoint folder

    Returns:
        dict: dictionary with all parameters
    """
    params = {}
    iteration_count = _read_best_iteration_number(filepath)

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
    params["uratio"] = int(param_string[5][6:])
    params["wd"] = float(param_string[6][2:])
    params["wu"] = float(param_string[7][2:])
    params["seed"] = int(param_string[8][4:])
    params["numlabels"] = int(param_string[9][9:])
    params["opt"] = param_string[10][3:]
    if len(param_string) > 11:
        if param_string[11] == "pretrained":
            params["pretrained"] = "pretrained"

    params["iterations"] = iteration_count
    return params


def clean_results_df(
    original_df, data_folder_name, sort_criterion="net", keep_per_class=False
):
    """Removing unnecessary columns to save into the csv file, sorting rows according to the sort_criterion, sorting colums according to the csv file format.

    Args:
        original_df ([df]): original dataframe to clean.
        data_folder_name ([str]): string containing experiment results
        sort_criterion (str, optional): Default criterion for rows sorting. Defaults to "net".
        keep_per_class (bool, optional): If True will not discard class-wise accuracy

    Returns:
        [cleaned outputdata]: [df]
    """
    if keep_per_class:
        new_df = original_df.drop(
            labels=[
                "batch_size",
                "seed",
                "use_train_model",
                "params",
                "macro avg",
                "weighted avg",
                "data_dir",
            ],
            axis=1,
        )
    else:
        dataset_name=original_df.index[0]
        
        if dataset_name == "ucm":
            new_df = original_df.drop(
                labels=[
                    "batch_size",
                    "seed",
                    "use_train_model",
                    "params",
                    "agricultural",
                    "airplane",
                    "baseballdiamond",
                    "beach",
                    "buildings",
                    "chaparral", 
                    "denseresidential",
                    "forest",
                    "freeway",
                    "golfcourse",
                    "harbor",
                    "intersection",
                    "mediumresidential",
                    "mobilehomepark",
                    "overpass",
                    "parkinglot",
                    "river",
                    "runway",
                    "sparseresidential",
                    "storagetanks",
                    "tenniscourt",
                    "macro avg",
                    "weighted avg",
                    "data_dir",
                ],
                axis=1,
            )
            
        else:
            new_df = original_df.drop(
                labels=[
                    "batch_size",
                    "seed",
                    "use_train_model",
                    "params",
                    "Forest",
                    "AnnualCrop",
                    "HerbaceousVegetation",
                    "Highway",
                    "Industrial",
                    "Pasture",
                    "PermanentCrop",
                    "River",
                    "Residential",
                    "SeaLake",
                    "macro avg",
                    "weighted avg",
                    "data_dir",
                ],
                axis=1,
            )    

    # Swap accuracy positions to sort it as in the final results file
    keys = new_df.columns.tolist()
    keys = keys[1:-1] + [keys[0]] + [keys[-1]]
    new_df = new_df.reindex(columns=keys)

    net = new_df["net"]
    if "pretrained" in new_df:
        # Removing unsorted and wrong pretrained column
        new_df = new_df.drop(labels=["pretrained"], axis=1)
        pretrained = np.array("True").repeat(len(net))
    else:
        pretrained = np.array("False").repeat(len(net))

    supervised = np.array(
        "False" if ("supervised" not in data_folder_name) else "True"
    ).repeat(len(net))

    # Adding new pretained and supervised columns
    new_df.insert(1, "supervised", supervised)
    new_df.insert(1, "pretrained", pretrained)

    # Returning new_df sorted by values according to the sort_criterion
    return new_df.sort_values(by=[sort_criterion], axis=0)

