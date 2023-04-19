import os
import urllib.request, json

from typing import List, Dict, Tuple, Callable, Optional, Any
import torch.cuda
import torchvision.transforms
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import ConcatDataset

# url = "https://storage.googleapis.com/kagglesdsdata/datasets/2310343/5229726/plant%20diseases%20cure/cure.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230415%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230415T155525Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=67295bd32ad3e66d630bb4868c93a8bb339dbf65262984914ff01203c8ebbf986654895b5b6c41d891cce19f92e170daa3e731c12d97fb45a70189efc459c9b6e0f2abdf7cf7e58c577f119f3c61f776bc71b6e9617540a5d000d444e9594b2bc9ffd384607808f209ba86790bfafa26f6191c76ed5c33b3f43e622425291a8577ca58c1acd4bd260e94c521a0dff449005d2a03e1a54da019406c9b88ede78b083499f3ca3d3603c9eb5443f7657bbfdf140acac9ce005615d95e888c41851524058edad6fd09da37efea4d6fff48f06f28d293cfe27227aab95bee06d51185429331eed4788cc60768627eaba31af0c3f389d225cfcfedd501ad31874c2671"
# with urllib.request.urlopen(url) as response:
#     json_obj = json.load(response)
#     json_obj_formatted = json.dumps(json_obj, indent=2)
#     print(json_obj_formatted)


def is_valid_file(filename: str) -> bool:
    return filename.lower().endswith(('.jpg', '.png'))


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.

    Change desired_classes list to load specific classes
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    desired_classes = ["apple_black_rot", "apple_cedar_rust", "apple_healthy", "apple_scab"]
    classes = [cls_name for cls_name in classes if cls_name in desired_classes]

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class CustomImageFolder(torchvision.datasets.DatasetFolder):
    """ Implementing custom ImageFolder class that overrides DatasetFolder methods, so it's possible to load only
    specific subdirectories(classes) of the directory instead of the whole directory.

    Enabled two valid extensions (.jpg, .png)
    """
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader,
                         extensions=('.jpg', '.png'), is_valid_file=is_valid_file)
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)


def train():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device)

    root_dir = "data/plant_dataset_original/plant_diseases_images"

    # apple_classes = ["apple_black_rot", "apple_cedar_rust", "apple_healthy", "apple_scab"]
    # sub_dirs = [os.path.join(root_dir, subdir) for subdir in apple_classes if os.path.isdir(os.path.join(root_dir, subdir))]

    transformations = transforms.Compose([
        torchvision.transforms.Resize((225, 225)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = CustomImageFolder(root=root_dir, loader=default_loader, transform=transformations)
    print(dataset)

train()