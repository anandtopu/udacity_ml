import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models


# Function to load and preprocess the data
def load_data(data_dir):
    trainTransforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    testTransforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 

    trainData = datasets.ImageFolder(data_dir + '/train', transform=trainTransforms)
    testData = datasets.ImageFolder(data_dir + '/test', transform=testTransforms)
    validData = datasets.ImageFolder(data_dir + '/valid', transform=testTransforms)

    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testData, batch_size=32)
    validLoader = torch.utils.data.DataLoader(validData, batch_size=32)
    
    return trainLoader, testLoader, validLoader, trainData, testData, validData


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(f'{image}' + '.jpg')

    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    pil_transform = transform(pil_image)
    
    # to convert to numpy array 
    array_image_transform = np.array(pil_transform)
    
    image_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)

    image_add_dim = image_tensor.unsqueeze_(0)
    
    return image_add_dim