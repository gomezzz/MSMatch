import imageio
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte,img_as_float32
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

class UCMDataset(torch.utils.data.Dataset):
    """UCM dataset."""

    def __init__(self, train, root_dir = "data/UCMerced_LandUse/Images/", transform=None, seed = 42):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.seed = seed
        self.size = [256,256]
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self._load_data()

    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed.
        """
        images = np.zeros([2100, self.size[0], self.size[1], 3],dtype="uint8")
        labels = []

        i = 0
        # read all the files from the image folder
        for item in tqdm(os.listdir(self.root_dir)):
            f = os.path.join(self.root_dir, item)
            if os.path.isfile(f):
                continue
            for subitem in os.listdir(f):
                sub_f = os.path.join(f, subitem)
                # a few images are a few pixels off, we will resize them
                image = imageio.imread(sub_f)
                if image.shape[0] != self.size[0] or image.shape[1] != self.size[1]:
                    # print("Resizing image...")
                    image = img_as_ubyte(
                        resize(image, (self.size[0], self.size[1]), anti_aliasing=True)
                    )
                images[i] = img_as_ubyte(image)
                i += 1
                labels.append(item)

        # convert to integer labels
        le = preprocessing.LabelEncoder()
        le.fit(np.sort(np.unique(labels)))
        labels = le.transform(labels)
        labels = np.asarray(labels)
        self.label_encoding = list(le.classes_) #remember label encoding

        # split into a train and test set as provided data is not presplit
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
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

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img ,self.targets[idx]