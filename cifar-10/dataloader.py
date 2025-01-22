from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from augment import RandAugment

def get_cifar10_dataset(root='../data', batch_size=512):
    # 데이터 로드 및 전처리
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_valid = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    N = 2; M = 14
    transform_train.transforms.insert(0, RandAugment(N, M))


    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader