import os
import torch
import torchvision.datasets as datasets


import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class PytorchStanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("Hugging Face datasets is not found. Please install it: pip install datasets")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        
        # Load dataset from Hugging Face - don't materialize all samples in memory
        self._dataset = load_dataset("tanganke/stanford_cars", split=self._split, cache_dir=root)
        
        # Extract class information
        self.classes = self._dataset.features['label'].names
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        item = self._dataset[idx]
        pil_image = item['image']
        target = item['label']
        
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(pil_image)  # or appropriate conversion
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
    
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        # Download is handled automatically by load_dataset
        pass


class Cars:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code

        self.train_dataset = PytorchStanfordCars(location, 'train', preprocess, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = PytorchStanfordCars(location, 'test', preprocess, download=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]
