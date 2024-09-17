import glob
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def xywh2xyxy(x: np.array):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


class FireSeriesDataset(Dataset):
    def __init__(self, root_dir, img_size=224, transform=None, crop_margin=1.2):
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize(
                        (img_size, img_size)
                    ),  # Resize to the desired img_size
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # ImageNet normalization
                ]
            )
        )
        self.sets = glob.glob(f"{root_dir}/**/*")
        random.shuffle(self.sets)
        self.img_size = img_size
        self.crop_margin = crop_margin

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        img_folder = self.sets[idx]
        img_list = glob.glob(f"{img_folder}/*.jpg")
        img_list.sort()

        images = [Image.open(file) for file in img_list]
        w, h = images[0].size

        # Collect labels for bounding boxes (assuming one label per image)
        labels = []
        for file in img_list:
            label_file = file.replace("images", "labels").replace(".jpg", ".txt")
            with open(label_file, "r") as f:
                lines = f.readlines()

            # Assuming the first line in each label file contains the necessary bounding box info
            labels.append(np.array(lines[0].split(" ")[1:5]).astype("float"))

        labels = np.array(labels)

        labels = xywh2xyxy(labels)

        x0, y0 = np.min(labels[:, :2], 0)
        x1, y1 = np.max(labels[:, 2:], 0)

        x0, y0, x1, y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        xc = x0 + (x1 - x0) / 2
        yc = y0 + (y1 - y0) / 2

        crop_size = max(x1 - x0, y1 - y0) * self.crop_margin

        if crop_size < self.img_size:
            crop_x0 = max(int(xc - self.img_size / 2), 0)
            crop_x1 = min(int(xc + self.img_size / 2), w)
            crop_y0 = max(int(yc - self.img_size / 2), 0)
            crop_y1 = min(int(yc + self.img_size / 2), h)

        else:
            crop_x0 = max(int(xc - crop_size / 2), 0)
            crop_x1 = min(int(xc + crop_size / 2), w)
            crop_y0 = max(int(yc - crop_size / 2), 0)
            crop_y1 = max(int(yc + crop_size / 2), h)

        img_sequence = []

        # Crop, resize, and transform each image in the sequence
        for im in images:
            cropped_image = im.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            if crop_size > self.img_size:
                cropped_image = cropped_image.resize((self.img_size, self.img_size))

            if self.transform:
                cropped_image = self.transform(cropped_image)

            img_sequence.append(cropped_image)

        # Stack the images into a tensor with shape (sequence_length, C, H, W)
        img_sequence = torch.stack(img_sequence, dim=0)

        # Return the sequence of images as a tensor and the corresponding label
        return img_sequence, int(img_folder.split("/")[-2])  # Adjust label as necessary
