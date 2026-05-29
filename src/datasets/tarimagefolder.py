import tarfile
import glob
from PIL import Image
from torch.utils.data import Dataset
import os

class TarImageFolder(Dataset):
    """
    Dataset that reads images from tar files, mimicking ImageFolder behavior.
    Maintains exact same ordering and indexing as ImageFolder.
    """
    def __init__(self, tar_dir, transform=None):
        self.transform = transform
        self.tar_files = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
        self.samples = []
        self.targets = []
        self.classes = []
        
        # Build class list from tar filenames (like ImageFolder does from folders)
        self.classes = [os.path.basename(tar).replace('.tar', '') 
                       for tar in self.tar_files]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Index all images in tars (maintains sorted order like ImageFolder)
        print(f"Indexing tar files from {tar_dir}...")
        for class_idx, tar_path in enumerate(self.tar_files):
            with tarfile.open(tar_path, 'r') as tar:
                # Get all image files, sorted (like ImageFolder)
                members = [m for m in tar.getmembers() if m.isfile() and 
                          (m.name.endswith('.JPEG') or m.name.endswith('.jpg') or m.name.endswith('.png'))]
                members = sorted(members, key=lambda x: x.name)
                
                for member in members:
                    self.samples.append((tar_path, member.name, class_idx))
                    self.targets.append(class_idx)
            
            if (class_idx + 1) % 100 == 0:
                print(f"Indexed {class_idx + 1}/{len(self.tar_files)} classes...")
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tar_path, member_name, class_idx = self.samples[idx]
        
        # Open tar and extract image
        with tarfile.open(tar_path, 'r') as tar:
            f = tar.extractfile(member_name)
            img = Image.open(f).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, class_idx

class TarImageFolderWithPaths(TarImageFolder):
    """Version that returns image, label, and path (like ImageFolderWithPaths)"""
    def __getitem__(self, idx):
        tar_path, member_name, class_idx = self.samples[idx]
        
        with tarfile.open(tar_path, 'r') as tar:
            f = tar.extractfile(member_name)
            img = Image.open(f).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Return path as: tar_filename/image_name
        path = f"{os.path.basename(tar_path)}/{member_name}"
        
        return img, class_idx, path
