# SSLRS
Semi-Supervised Learning Remote Sensing

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

The Semi-Supervised Learning Remote Sensing (SSLRS) project aims to apply the state of the art of Semi-Supervised learning techniques to land-use and land-cover classification problems. Currently, the repository includes an implementation of [FixMatch](https://arxiv.org/abs/2001.07685) for the training of [EfficientNet](https://arxiv.org/abs/1905.11946) Convolutional Neural Networks. The code exploits and the extends the [FixMatch-pytorch](https://github.com/LeeDoYup/FixMatch-pytorch) implementation based on [PyTorch](https://pytorch.org/). Compared to the original repository, this repository includes both the RGB and the Multi-Spectral (MS) versions of [EuroSAT](https://arxiv.org/abs/1709.00029) dataset.

### Built With

* [PyTorch](https://pytorch.org/)
* [conda](https://docs.conda.io/en/latest/)


<!-- GETTING STARTED -->
## Getting Started

This is a brief example of setting up SSLRS.

### Prerequisites

We recommend using [conda](https://docs.conda.io/en/latest/) to set-up your environment. This will also automatically set up CUDA and the cudatoolkit for you, enabling the use of GPUs for training, which is recommended.


* [conda](https://docs.conda.io/en/latest/), which will take care of all requirements for you. For a detailed list of required packages, please refer to the [conda environment file](https://github.com/gomezzz/SSLRS/blob/main/environment.yml).

### Installation

1. Get [miniconda](https://docs.conda.io/en/latest/miniconda.html) or similar
2. Clone the repo
   ```sh
   git clone https://github.com/gomezzz/SSLRS.git
   ```
3. Setup the environment. This will create a conda environment called `torchmatch`
   ```sh
   conda env create -f environment.yml
   ```

### Set-up datasets
To launch the training on EuroSAT (rgb or MS), it is necessary to download the corresponding datasets. The `root_dir` variable in the corresponding `datasets/eurosat_dataset.py` and `datasets/eurosat_rgb_dataset.py` files shall be adjusted according to the dataset path. 
  
<!-- Content of Repo -->
## Content of Repository

The repository is structured as follows: 

- **datasets**: including the Semi-Supervised learning datasets usable for training, and augmentation files. To add a new dataset, a new `.py` file shall be added.
- **external/visualizations**: containing tools for extract saliency information of trained models. We exploited the code include in the `src` directory of [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) repository.

- **models**: including the neural networks models used for training.
- **notebooks**: containing some jupyter notebooks used to extract saliency of images, collect training results, showing augmentation effects on images and providing additional functionalities. To be able to use the notebooks, it is necessary to install [Jupyter](https://jupyter.org/).
- **runscripts**: including scripts used to train the networks.
- **utils.py**: including functions used for notebooks.
- **train_utils.py**: providing utils for training.
- **train.py**: train script.
- **eval.py**: script for evaluating a trained network.
- **LICENSE**: license file
- **README**: this file.
- **environment.yml**: environment file. 


<!-- USAGE EXAMPLES -->
## Usage

### Train a model

To train a model on EuroSAT RGB by using EfficientNet B0 from scratch,  you can use: 
```
python train_model.py --dataset eurosat_rgb --net efficientnet_b0
```

`--net ` can be used to specify the EfficientNet model, whilst `--dataset` can be used to specify the dataset. Use `eurosat_rgb` for EuroSAT RGB and `eurosat_ms` for EuroSAT MS dataset.

Instead of starting the training from scratch, it is possible exploit a model pretrained on ImageNet. To do it,  you can use: 
```
python train_model.py --dataset eurosat_rgb --net efficientnet_b0 --pretrained
```

`--net ` can be used to specify the EfficientNet model, whilst `--dataset` can be used to specify the dataset. Use `eurosat_rgb` for EuroSAT RGB and `eurosat_ms` for EuroSAT MS dataset.

Information on additional flags can be obtained by typing:
```
python train_model.py --help
```

For additional information on training, including the use of single/multiple GPUs, please refer to [FixMatch-pytorch](https://github.com/LeeDoYup/FixMatch-pytorch).

### Evaluate a model

To evaluate a trained model on a target dataset, you can use:

```
python eval.py --load_path [LOAD_PATH] --dataset [DATASET] --net [NET]
```

where `LOAD_PATH` is the path of the trained model (`.pth` file), `DATASET` is the target dataset, `NET` is the network model used during the training.


## Roadmap

See the [open issues](https://github.com/gomezzz/SSLRS/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

The project is open to community contributions. Feel free to open an [issue](https://github.com/gomezzz/SSLRS/issues) or write us an email if you would like to discuss a problem or idea first.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See [LICENSE](https://github.com/esa/torchquad/blob/main/LICENSE) for more information.


<!-- FAQ -->
## FAQ 

  1. Q: `Error enabling CUDA. cuda.is_available() returned False. CPU will be used.`  <br/>A: This error indicates that no CUDA-compatible GPU could be found. Either you have no compatible GPU or the necessary CUDA requirements are missing. Using `conda`, you can install them with `conda install cudatoolkit`. For more detailed installation instructions, please refer to the [PyTorch documentation](https://pytorch.org/get-started/locally/).




<!-- CONTACT -->
## Contact 

Created by ESA's [Advanced Concepts Team](https://www.esa.int/gsp/ACT/index.html)

- Pablo Gómez - `pablo.gomez at esa.int`
- Gabriele Meoni - `gabriele.meoni at esa.int`

Project Link: [https://github.com/esa/torchquad](https://github.com/esa/torchquad)



<!-- ACKNOWLEDGEMENTS 
This README was based on https://github.com/othneildrew/Best-README-Template
-->

