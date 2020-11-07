import argparse
import os
from torch.utils.data import Dataset
from skimage import io
import torch
import torchvision
import torchvision.transforms as transforms
from instance_selection import select_instances

class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        filenames = os.listdir(self.root_dir)
        self.filepaths = [os.path.join(self.root_dir, filename) for filename in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.filepaths[idx]
        image = io.imread(path)
        sample = {'image': image, 'path': path}

        if self.transform:
            sample = self.transform(sample)

        return sample

def selecter(target_folder):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
    dataset = CustomDataset(target_folder)
    instance_selected_dataset = select_instances(dataset, retention_ratio=50)

def main():
    parser = argparse.ArgumentParser(
        description='Select a folder with images to generate a subsample of selected ones.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--target-folder',      help='Target the folder containing the images to project from', dest='target_folder', required=True)
    selecter(**vars(parser.parse_args()))

if __name__ == "__main__":
    main()