from torch.utils.data import Dataset
from torchvision import transforms

class ImagePoseDataset(Dataset):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]

        image = self.transform(image)

        return image, pose