import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.Dataset import CASIADataset
# from datasets.patch_Dataset import PATCHDataset
from sklearn.model_selection import train_test_split

def load_dataloader(data_path, split_ratio, batch_size, img_size, num_workers) :
    Tp_image_path = os.path.join(data_path, 'CASIA2.0_revised/Tp')
    Tp_label_path = os.path.join(data_path, 'CASIA2.0_Groundtruth')
    Au_image_path = os.path.join(data_path, 'CASIA2.0_revised/Au')

    Tp_images = sorted([Tp_image_path + '/' + x for x in os.listdir(Tp_image_path)]) #[:100]
    Tp_labels = sorted([Tp_label_path + '/' + x for x in os.listdir(Tp_label_path)]) #[:100]
    Au_images = sorted([Au_image_path + '/' + x for x in os.listdir(Au_image_path)]) #[:100]
    Au_labels = ['Au' for _ in range(len(Au_images))]

    Total_images = Tp_images + Au_images
    Total_labels = Tp_labels + Au_labels

    train_x, test_x, train_y, test_y = train_test_split(Total_images, Total_labels, test_size=split_ratio, shuffle=True)

    cnt = 0
    for label_ in train_y :
        if label_ == 'Au' :
            cnt += 1
    print("#train Tp = ", len(train_y) - cnt)
    print("#train Au = ", cnt)
    cnt = 0
    for label_ in test_y :
        if label_ == 'Au' :
            cnt += 1
    print("#test Tp = ", len(test_y) - cnt)
    print("#test Au = ", cnt)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10, expand=False),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = CASIADataset(train_x, train_y, img_size, train_transform)
    test_dataset = CASIADataset(test_x, test_y, img_size, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print('create data loader success')

    return train_loader, test_loader