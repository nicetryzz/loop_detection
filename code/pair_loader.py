import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
import cv2

class ImagePairDataset(Dataset):
    def __init__(self, txt_file, folder_path, transform=None, need_translate = True):
        # Open the txt file and read the lines
        with open(txt_file, 'r') as f:
            self.lines = f.readlines()
        # Store the transform if given
        self.transform = transform
        self.folder_path = folder_path
        self.need_translate = need_translate
    
    def __len__(self):
        # Return the number of pairs in the dataset
        return len(self.lines)
    
    def __getitem__(self, idx):
        # Get the line at the given index
        line = self.lines[idx]
        line = line.replace(" ", "")
        img1_path, img2_path, label = line.split(",")
        label = int(label.replace("\n",""))
        # Load the images using PIL
        if self.need_translate:
            img1 = cv2.imread(os.path.join(self.folder_path, img1_path), cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(os.path.join(self.folder_path, img2_path), cv2.IMREAD_UNCHANGED)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BayerBG2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BayerBG2RGB)
        else:
            img1 = Image.open(os.path.join(self.folder_path, img1_path))
            img2 = Image.open(os.path.join(self.folder_path, img2_path))
        # Apply the transform if given
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # Return the image pair as a tuple
        return (img1, img2, label)

def get_data_loader(folder_path,batch_size,split=True,need_translate = True):
    
    # Create instances of the datasets for different txt files
    transform = None
        
    transforms_list = [
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    transform = transforms.Compose(transforms_list)
    
    datasets = []
    
    txt_path = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            txt_path.append(os.path.join(folder_path, file))
    
    for path in txt_path:
        datasets.append(ImagePairDataset(path, folder_path=folder_path, transform=transform, need_translate=need_translate))
    
    # Concatenate the datasets into a single dataset and spilit
    dataset = ConcatDataset(datasets)
    if split:
        train_size = int(len(dataset)*0.8)
        valid_size = int(len(dataset)-train_size)
        train_subset, valid_subset = random_split(dataset, [train_size, valid_size])

        # Create data loaders for each subset with the desired batch size and shuffle option
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
        
        return train_loader, valid_loader
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return data_loader