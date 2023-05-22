import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os

class ImagePairDataset(Dataset):
    def __init__(self, txt_file, folder_path, transform=None):
        # Open the txt file and read the lines
        with open(txt_file, 'r') as f:
            self.lines = f.readlines()
        # Store the transform if given
        self.transform = transform
        self.folder_path = folder_path
    
    def __len__(self):
        # Return the number of pairs in the dataset
        return len(self.lines)
    
    def __getitem__(self, idx):
        # Get the line at the given index
        line = self.lines[idx]
        img1_path, img2_path, label = line.split(", ")
        label = int(label.replace("\n",""))
        # Load the images using PIL
        img1 = Image.open(os.path.join(self.folder_path, img1_path))
        img2 = Image.open(os.path.join(self.folder_path, img2_path))
        # Apply the transform if given
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # Return the image pair as a tuple
        return (img1, img2, label)

def get_data_loader(folder_path):
    
    # Create instances of the datasets for different txt files
    transform = None
        
    transforms_list = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    transform = transforms.Compose(transforms_list)
    
    datasets = []
    
    txt_path = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            txt_path.append(os.path.join(folder_path, file))
    
    for path in txt_path:
        datasets.append(ImagePairDataset(path, folder_path=folder_path, transform=transform))
    
    # Concatenate the datasets into a single dataset and spilit
    dataset = ConcatDataset(datasets)

    train_size = int(len(dataset)*0.8)
    valid_size = int(len(dataset)*0.1)
    test_size = int(len(dataset)*0.1)
    train_subset, valid_subset, test_subset = random_split(dataset, [train_size, valid_size, test_size])

    # Create data loaders for each subset with the desired batch size and shuffle option
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=4, shuffle=False)
    
    return train_loader, valid_loader, test_loader