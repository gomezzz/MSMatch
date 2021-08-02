[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msmatch-semi-supervised-multispectral-scene/image-classification-on-eurosat)](https://paperswithcode.com/sota/image-classification-on-eurosat?p=msmatch-semi-supervised-multispectral-scene) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msmatch-semi-supervised-multispectral-scene/scene-classification-on-uc-merced-land-use)](https://paperswithcode.com/sota/scene-classification-on-uc-merced-land-use?p=msmatch-semi-supervised-multispectral-scene)

# MSMatch
Semi-Supervised Multispectral Scene Classification with Few Labels

<!--
*** Based on https://github.com/othneildrew/Best-README-Template
-->



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#set-up-datasets">Set-up datasets</a></li>
      </ul>
    </li>
    <li><a href="#content-of-repository">Content of Repository</a></li>
    <li><a href="#usage">Usage</a>
    <ul>
        <li><a href="#train-a-model">Train a model</a></li>
        <li><a href="#evaluate-a-model">Evaluate a model</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#set-up-datasets">Set-up datasets</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#FAQ">FAQ</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is the code for the paper *"MSMatch: Semi-Supervised Multispectral Scene Classification with Few Labels"* by Pablo Gómez and Gabriele Meoni, which aims to apply the state of the art of semi-supervised learning techniques to land-use and land-cover classification problems. 
Currently, the repository includes an implementation of [FixMatch](https://arxiv.org/abs/2001.07685) for the training of [EfficientNet](https://arxiv.org/abs/1905.11946) Convolutional Neural Networks. The code builds on and extends the [FixMatch-pytorch](https://github.com/LeeDoYup/FixMatch-pytorch) implementation based on [PyTorch](https://pytorch.org/). Compared to the original repository, this repository includes code to work with both the RGB and the multispectral (MS) versions of [EuroSAT](https://arxiv.org/abs/1709.00029) dataset and the [UC Merced Land Use ](http://weegee.vision.ucmerced.edu/datasets/landuse.html) (UCM) dataset.

### Built With

* [PyTorch](https://pytorch.org/)
* [conda](https://docs.conda.io/en/latest/)
* [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
* [albumentations](https://github.com/albumentations-team/albumentations)
* imageio, numpy, pandas

<!-- GETTING STARTED -->
## Getting Started

This is a brief example of setting up MSMatch.

### Prerequisites

We recommend using [conda](https://docs.conda.io/en/latest/) to set-up your environment. This will also automatically set up CUDA and the cudatoolkit for you, enabling the use of GPUs for training, which is recommended.


* [conda](https://docs.conda.io/en/latest/), which will take care of all requirements for you. For a detailed list of required packages, please refer to the [conda environment file](https://github.com/gomezzz/MSMatch/blob/main/environment.yml).

### Installation

1. Get [miniconda](https://docs.conda.io/en/latest/miniconda.html) or similar
2. Clone the repo
   ```sh
   git clone https://github.com/gomezzz/MSMatch.git
   ```
3. Setup the environment. This will create a conda environment called `torchmatch`
   ```sh
   conda env create -f environment.yml
   ```

### Set up datasets
To launch the training on EuroSAT (rgb or MS), it is necessary to download the corresponding datasets. The `root_dir` variable in the corresponding `datasets/eurosat_dataset.py` and `datasets/eurosat_rgb_dataset.py` files shall be adjusted according to the dataset path. 
  
<!-- Content of Repo -->
## Content of Repository

The repository is structured as follows: 

- **datasets**: contains the semi-supervised learning datasets usable for training, and augmentation code. To add a new dataset, a new class similar to, e.g., `eurosat_rgb.py`needs to be added.
- **external/visualizations**: contains tools to create visualizations of trained models. We utilized the code from the `src` directory of [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) repository and slightly adapted it.
- **models**: contains the neural networks models used for training.
- **notebooks**: contains some jupyter notebooks used to create paper figures, collect training results, showing augmentation effects on images and provide additional functionalities. To be able to use the notebooks, it is necessary to additionally install [Jupyter](https://jupyter.org/).
- **runscripts**: includes bash scripts used to train the networks.
- **`utils.py`**: some utility functions.
- **`train_utils.py`**: providing utils for training.
- **`train.py`**: main train script.
- **`eval.py`**: main script for evaluating a trained network.
- **`environment.yml`**: conda environment file describing dependencies. 


<!-- USAGE EXAMPLES -->
## Usage

### Train a model

To train a model on EuroSAT RGB by using EfficientNet B0 from scratch,  you can use: 
```
python train.py --dataset eurosat_rgb --net efficientnet-b0
```

`--net ` can be used to specify the EfficientNet model, whilst `--dataset` can be used to specify the dataset. Use `eurosat_rgb` for EuroSAT RGB, `eurosat_ms` for EuroSAT MS, and `ucm` for UCM dataset.

Instead of starting the training from scratch, it is possible exploit a model pretrained on ImageNet. To do it,  you can use: 
```
python train.py --dataset eurosat_rgb --net efficientnet-b0 --pretrained
```

Information on additional flags can be obtained by typing:
```
python train.py --help
```

For additional information on training, including the use of single/multiple GPUs, please refer to [FixMatch-pytorch](https://github.com/LeeDoYup/FixMatch-pytorch).

### Evaluate a model

To evaluate a trained model on a target dataset, you can use:

```
python eval.py --load_path [LOAD_PATH] --dataset [DATASET] --net [NET]
```

where `LOAD_PATH` is the path of the trained model (`.pth` file), `DATASET` is the target dataset, `NET` is the network model used during the training.


## Roadmap

See the [open issues](https://github.com/gomezzz/MSMatch/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

The project is open to community contributions. Feel free to open an [issue](https://github.com/gomezzz/MSMatch/issues) or write us an email if you would like to discuss a problem or idea first.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See [LICENSE](https://github.com/gomezzz/MSMatch/blob/main/LICENSE) for more information.

<!-- CONTACT -->
## Contact 

Created by ESA's [Advanced Concepts Team](https://www.esa.int/gsp/ACT/index.html)

- Pablo Gómez - `pablo.gomez at esa.int`
- Gabriele Meoni - `gabriele.meoni at esa.int`

Project Link: [https://www.esa.int/gsp/ACT/projects/semisupervised/](https://www.esa.int/gsp/ACT/projects/semisupervised/)



<!-- ACKNOWLEDGEMENTS 
This README was based on https://github.com/othneildrew/Best-README-Template
-->
