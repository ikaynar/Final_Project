#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[7]:


# Imports here

get_ipython().system('pip install torchsummary')
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import torch.nn as nn
from torchsummary import summary 
import copy
#from torchsummary import summary


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[8]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[9]:


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transform= transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])


# TODO: Load the datasets with ImageFolder
train_dataset=ImageFolder(train_dir, transform=data_transforms)
val_dataset=ImageFolder(valid_dir, transform=data_transforms)
test_dataset=ImageFolder(test_dir, transform=data_transforms)
image_datasets=(train_dataset, val_dataset, test_dataset)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader=DataLoader(image_datasets[0])
val_loader=DataLoader(image_datasets[1])
test_loader=DataLoader(image_datasets[2])
dataloaders=(train_loader, val_loader, test_loader)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[10]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name=json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# <font color='red'>**Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.</font>

# In[ ]:


# TODO: Build and train your network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer,train_dataloader, val_loader, num_epochs=25):
    
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0

    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
            
            
        running_loss = 0.0
        val_loss=0.0
            
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
                
                # zero the parameter gradients
            optimizer.zero_grad()
                
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                
                # statistics
            running_loss += loss.item() * inputs.size(0)
                
        epoch_loss = running_loss / len(train_dataloader)
        
        print('Epoch LOSS: {}'.format(epoch_loss))
        
        model.eval()
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            optimizer.zero_grad()
                
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

        total_val_loss = val_loss / len(val_loader)
        
        print('Epoch LOSS: {}'.format(epoch_loss))

            
    return model

model_ft = models.vgg16(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False


num_ftrs = model_ft.classifier[6].in_features

for param in model_ft.classifier[6].parameters():
    param.requires_grad=True

model_ft.classifier[6]=nn.Linear(num_ftrs, 102)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.classifier[6].parameters(), lr=0.001)


# In[ ]:


summary(model_ft, (3,224,224))


# In[ ]:


#summary(model_ft.cuda(), (3, 224, 224))


# In[ ]:


train_model(model_ft, criterion, optimizer_ft, train_loader, val_loader, num_epochs=25)


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[42]:


# TODO: Do validation on the test set
def test(model, test_loader):
    
    model.eval()
    running_corrects
    for images, labels in test_loader:
        
        outputs=model(images)
        _, preds=torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds==labels.data)
    total_acc=running_corrects.double() / len(test_loader)
    
    return total_acc
        


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[43]:


# TODO: Save the checkpoint 

import pickle
with open("./class_to_idx", "wb+") as fd:
    pickle.dump(train_dataset.class_to_idx, fd)
torch.save(model_ft.state_dict(), "./model_saved.pth")


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[44]:


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(path_to_weights):
    model = models.vgg16()
    in_fts = model.classifier[6].in_features
    model.classifier[6]=nn.Linear(in_fts,102)
    model.load_state_dict(torch.load(path_to_weights))
    return model

load_model("./model_saved.pth")


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[45]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #img=cv2.imread(image)
    #img=cv2.resize(img, (256,-1))
     
    img=image
    scale_percent = 256/img.shape[0]
    width = int(img.shape[1]*scale_percent/100)
    height = int(img.shape[0]*scale_percent/100)
    dim = (witdh, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    h_height = resized.shape[0] // 2
    h_width = resized.shape[1] // 2
    
    cropped= resized[h_height-112:h_height+112, h_width-112:h_width+112,:]
    cropped=cropped/255
    
    means=[0.485, 0.456, 0.406]
    sds= [0.229, 0.224, 0.225]
    
    cropped[:,:,0] = (cropped[:,:,0] - means[0])/sds[0]
    cropped[:,:,1] = (cropped[:,:,1] - means[1])/sds[1]
    cropped[:,:,2] = (cropped[:,:,2] - means[2])/sds[2]
    
    return cropped
    
    # TODO: Process a PIL image for use in a PyTorch model


# In[46]:


def preprocess(img_path,k):
    transfer_test_transforms = transforms.Compose([
        Resize((256,-1)),
        CenterCrop((224,224)).
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = Image.open(img_path)
    img = transfer_test_transforms(img).reshape(-1,3,224,224).cuda()
    
    labels_dict = _get_label_names("/data/landmark_images/train/")
    
    model_transfer.eval()
    preds = model_transfer(img)
    results_num = torch.topk(preds, k)[1].cpu().numpy()[0]
    results=[]
    
    for rank in result_num:
        results.append(labels_dict[rank])
    
    return results


# In[47]:


def preprocess(img):
    transfer_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    
    img = transfer_test_transforms(img)
    
    return img


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[48]:


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


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[49]:


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
    # TODO: Implement the code to predict the class from an image file


# In[50]:


predict("flowers/train/1/image_06734.jpg", model_ft)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[51]:


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


# In[52]:


list_of_names,list_of_probs=predict_and_plot("flowers/train/1/image_06734.jpg", model_ft)


# <font color='red'>**Reminder for Workspace users:** If your network becomes very large when saved as a checkpoint, there might be issues with saving backups in your workspace. You should reduce the size of your hidden layers and train again. 
#     
# We strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.</font>

# In[53]:


# TODO remove .pth files or move it to a temporary `~/opt` directory in this Workspace

get_ipython().getoutput('jupyter nbconvert *.ipynb')


# In[ ]:



