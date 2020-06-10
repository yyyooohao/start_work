from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

tf = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, root, img_size):
        self.dataset = []

        self.root_dir = f"{root}/{img_size}"
        with open(f"{self.root_dir}/positive.txt", "r") as f:
            self.dataset.extend(f.readlines())
        with open(f"{self.root_dir}/negative.txt", "r") as f:
            self.dataset.extend(f.readlines())
        with open(f"{self.root_dir}/part.txt", "r") as f:
            self.dataset.extend(f.readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        strs = data.split()
        if strs[1] == "1":
            img_path = f"{self.root_dir}/positive/{strs[0]}"
        elif strs[1] == "2":
            img_path = f"{self.root_dir}/part/{strs[0]}"
        else:
            img_path = f"{self.root_dir}/negative/{strs[0]}"

        img_data = tf(Image.open(img_path))

        c, x1, y1, x2, y2 = float(strs[1]), float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])
        return img_data, np.array([c, x1, y1, x2, y2], dtype=np.float32)

if __name__=="__main__":
    dataset = MyDataset("F:\mtcnn_data", 12)
    print(dataset[0])











