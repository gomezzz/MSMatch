import imageio
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm


class EurosatDataset(torch.utils.data.Dataset):
    """Eurosat dataset"""

    def __init__(
        self, train, root_dir="data/EuroSATallBands/", transform=None, seed=42
    ):
        """
        Args:
            train (bool): If true returns training set, else test
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            seed (int): seed used for train/test split
        """
        self.seed = seed
        self.size = [64, 64]
        self.num_channels = 13
        self.num_classes = 10
        self.root_dir = root_dir
        self.transform = transform
        self.test_ratio = 0.1
        self.train = train
        self.N = 27000
        self._load_data()

    def _normalize_to_0_to_1(self, img):
        """Normalizes the passed image to 0 to 1

        Args:
            img (np.array): image to normalize

        Returns:
            np.array: normalized image
        """
        img = img + np.minimum(0, np.min(img))  # move min to 0
        img = img / np.max(img)  # scale to 0 to 1
        return img

    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed. By default resized to 256,256
        """
        images = np.zeros([self.N, self.size[0], self.size[1], 13], dtype="uint8")
        labels = []
        filenames = []

        i = 0
        # read all the files from the image folder
        for item in tqdm(os.listdir(self.root_dir)):
            f = os.path.join(self.root_dir, item)
            if os.path.isfile(f):
                continue
            for subitem in os.listdir(f):
                sub_f = os.path.join(f, subitem)
                filenames.append(sub_f)
                # a few images are a few pixels off, we will resize them
                image = imageio.imread(sub_f)
                if image.shape[0] != self.size[0] or image.shape[1] != self.size[1]:
                    image = resize(
                        image, (self.size[0], self.size[1]), anti_aliasing=True
                    )
                images[i] = img_as_ubyte(self._normalize_to_0_to_1(image))
                i += 1
                labels.append(item)

        labels = np.asarray(labels)
        filenames = np.asarray(filenames)

        # sort by filenames
        images = images[filenames.argsort()]
        labels = labels[filenames.argsort()]

        # convert to integer labels
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)
        self.label_encoding = list(le.classes_)  # remember label encoding

        # split into a train and test set as provided data is not presplit
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=labels,
        )

        if self.train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]
