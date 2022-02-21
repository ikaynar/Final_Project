import numpy as np
import matplotlib.pyplot as plt
import torch

def load_model(path_to_model):
    return torch.load(path_to_model)

from PIL import Image
def preprocess(img):
    transfer_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    
    img = transfer_test_transforms(img)
    
    return img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
     
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    img = preprocess(img)
    #img = np.swapaxes(img, 0,2)
    #img = torch.tensor(img)
    #print(img.shape)
    #imshow(img)
    img = img.reshape(1,3,224,224)
    img =img.cuda()
    model.eval()
    preds = model(img)
    results_num = torch.topk(preds, topk)[1].cpu().numpy()[0]
    
    return results_num,preds


# TODO: Display an image along with the top 5 classes
import json

def predict_and_plot(path_to_image, model):
    index_to_class = {value:key for key, value in train_dataset.class_to_idx.items()}
    index_to_class
    with open("cat_to_name.json", "rb") as fd:
        name_dict=json.load(fd)

    top_idxs,preds= predict(path_to_image, model_ft)
    softmaxed_preds = nn.functional.softmax(preds)[0,:].detach().cpu().numpy()
    print(softmaxed_preds.shape)
    list_of_names=[]
    list_of_probs=[]
    for idx in top_idxs:
        class_label=index_to_class[idx]
        prediction_name=name_dict[class_label]
        prediction_val=softmaxed_preds[idx]
        
        list_of_names.append(prediction_name)
        list_of_probs.append(prediction_val)
    
    fig, ax=plt.subplots()
    img = Image.open(path_to_image)
    img=preprocess(img)

    imshow(img)
    ax.barh(list_of_names,list_of_probs)    
    
    return list_of_names,list_of_probs
