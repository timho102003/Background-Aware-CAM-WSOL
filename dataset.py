from curses import meta
import os
import cv2
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms as T


class CUB200(Dataset):
    def __init__(self, data_dir="./dataset/CUB200/images", data_list="./dataset/CUB200/lists/train.txt", transform=None) -> None:
        super(CUB200, self).__init__()
        self.txt_file = data_list
        self.data_dir = data_dir
        self.transform = transform
        with open(self.txt_file, "r") as f:
            self.data_list = f.readlines()
            self.data_list = [data.strip() for data in self.data_list]

    def __getitem__(self, index):
        imgname = self.data_list[index]
        target = int(imgname.split(".")[0])-1
        imgname = os.path.join(self.data_dir, imgname)
        img = cv2.imread(imgname)
        height, width, _ = img.shape
        if self.transform:
            img = self.transform(img)
        target = torch.tensor([target], dtype=torch.int64)
        meta_info = {"filename": imgname, "size": (height, width)}
        # print(meta_info)
        return img, target, meta_info
    
    def __len__(self):
        return len(self.data_list)


class CUB200Dataset(LightningDataModule):
    def __init__(self, data_dir: str="./dataset/CUB200/images", batch_size: int=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_txt = "./dataset/CUB200/lists/train.txt"
        self.test_txt = "./dataset/CUB200/lists/test.txt"
        self.train_transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.infer_transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 256)),
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.dims = (3, 224, 224)
        self.num_classes = 10

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cub200_full = CUB200(data_dir=self.data_dir,
                                 data_list=self.train_txt,
                                 transform=self.train_transform)
            train_num = int(len(cub200_full) * 0.9)
            self.cub200_train, self.cub200_val = random_split(
                cub200_full, [train_num, len(cub200_full) - train_num])

        # Assign test dataset for use in dataloader(s)
        if stage in ["test", "predict"] or stage is None:
            self.cub200_test = CUB200(data_dir=self.data_dir,
                                      data_list=self.train_txt, 
                                      transform=self.infer_transform)

    def train_dataloader(self):
        return DataLoader(self.cub200_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cub200_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.cub200_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.cub200_test, batch_size=1, shuffle=False)
