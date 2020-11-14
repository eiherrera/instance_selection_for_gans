import argparse
import os
from shutil import copyfile
from torch.utils.data import Dataset
from PIL import Image
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
        image = Image.open(path).convert('RGB')
        sample = {'image': image, 'path': path}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def selecter(target_folder, new_folder, retention_ratio=50):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])
    dataset = CustomDataset(target_folder, transform=transform)
    instance_selected_dataset = select_instances(dataset, retention_ratio=retention_ratio)
    for el in instance_selected_dataset:
        suffix = el['path'].split('/')[-1]
        new_path = os.path.join(new_folder, suffix)
        copyfile(el['path'], new_path)

def main():
    parser = argparse.ArgumentParser(
        description='Select a folder with images to generate a subsample of selected ones.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset',            help='Folder containing the dataset', dest='target_folder', required=True)
    parser.add_argument('--new-folder',         help='Folder to store the filtered images', dest='new_folder', required=True)
    parser.add_argument('--retention-ratio',    help='Percentage of images to keep after filter', dest='retention_ratio', type=int, default=50)

    selecter(**vars(parser.parse_args()))

if __name__ == "__main__":
    main()