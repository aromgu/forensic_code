import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.Dataset import CASIADataset
from sklearn.model_selection import train_test_split

def load_dataloader(data_path, split_ratio, batch_size, img_size) :
    Tp_image_path = os.path.join(data_path, 'CASIA2.0_revised/Tp')
    Tp_label_path = os.path.join(data_path, 'CASIA2.0_Groundtruth')

    Tp_images = sorted([Tp_image_path + '/' + x for x in os.listdir(Tp_image_path)])[:10]
    Tp_labels = sorted([Tp_label_path + '/' + x for x in os.listdir(Tp_label_path)])[:10]

    train_x, test_x, train_y, test_y = train_test_split(Tp_images, Tp_labels, test_size=split_ratio, shuffle=True)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, expand=False),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = CASIADataset(train_x, train_y, img_size, train_transform)
    test_dataset = CASIADataset(test_x, test_y, img_size, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print('create data loader success')

    return train_loader, test_loader