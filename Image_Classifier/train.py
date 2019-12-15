import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from utility_functions import load_data
from functions import build_classifier, validation, train_model, test_model, save_model, load_checkpoint

command_parser = argparse.ArgumentParser(description='Train neural network.')

# ../ImageClassifierflowers
command_parser.add_argument('data_directory', action = 'store',
                    help = 'Enter path to training data.')

command_parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg11',
                    help= 'Enter pretrained model to use; this classifier can currently work with\
                           VGG and Densenet architectures. The default is VGG-11.')

command_parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

command_parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001.')

command_parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.05,
                    help = 'Enter dropout for training the model, default is 0.05.')

command_parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 500,
                    help = 'Enter number of hidden units in classifier, default is 500.')

command_parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2,
                    help = 'Enter number of epochs to use during training, default is 1.')

command_parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')

results = command_parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
gpu_mode = results.gpu

# Load and preprocess data 
trainLoader, testLoader, validLoader, trainData, testData, validData = load_data(data_dir)

# Load pretrained model
pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

# Build and attach new classifier
input_units = model.classifier[0].in_features
build_classifier(model, input_units, hidden_units, dropout)

# Recommended to use NLLLoss when using Softmax
criterion = nn.NLLLoss()
# Using Adam optimiser which makes use of momentum to avoid local minima
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# to tain model
model, optimizer = train_model(model, epochs,trainLoader, validLoader, criterion, optimizer, gpu_mode)

# to test model
test_model(model, testLoader, gpu_mode)
# Save model
save_model(model, trainData, optimizer, save_dir, epochs)