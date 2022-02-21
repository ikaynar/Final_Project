from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

def load_data(data_dir):
    
    data_transforms= transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_dir=data_dir +"/train"
    valid_dir=data_dir +"/valid"
    test_dir=data_dir +"/test"
    
    train_dataset = ImageFolder(train_dir, transform=data_transforms)
    val_dataset = ImageFolder(valid_dir, transform=data_transforms)
    test_dataset = ImageFolder(test_dir, transform=data_transforms)
    image_dataset = (train_dataset, val_dataset, test_dataset)
    
    
    #img = transfer_test_transforms(img)
    
    return 