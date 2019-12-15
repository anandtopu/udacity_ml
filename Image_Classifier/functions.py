import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# Function to build new classifier
def build_classifier(model, input_units, hidden_units, dropout):
    # Weights of pretrained model are frozen so we don't backprop through/update them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(
        OrderedDict([('fc1', nn.Linear(input_units, hidden_units)),
                     ('relu', nn.ReLU()), ('dropout1', nn.Dropout(dropout)),
                     ('fc2',
                      nn.Linear(hidden_units, 102)), ('output',
                                                      nn.LogSoftmax(dim=1))]))

    # Replacing the pretrained classifier with the one above
    model.classifier = classifier
    return model


def validation(model, validLoader, criterion):
    valid_loss = 0
    accuracy = 0

    # choose cuda
    model.to('cuda')

    for I, (images, labels) in enumerate(validLoader):

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)

        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)

        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


def train_model(model, epochs, trainLoader, validLoader, criterion, optimizer,
                gpu_mode):
    steps = 0
    print_interval = 40

    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
    #model.to('cuda')

    for epoch in range(epochs):
        running_loss = 0

    for I, (inputs, labels) in enumerate(trainLoader):
        steps += 1

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_interval == 0:
            model.eval()

            with torch.no_grad():
                valid_loss, accuracy = validation(model, validLoader,
                                                  criterion)

            print(f"Epoch No. : {epoch+1}, \
            Training Loss: {round(running_loss/print_interval,3)} \
            Validation Loss: {round(valid_loss/len(validLoader),3)} \
            Validation Accuracy: {round(float(accuracy/len(validLoader)),3)}")

            running_loss = 0

            model.train()

    return model, optimizer


def test_model(model, testloader, gpu_mode):
    correct = 0
    total = 0

    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):

            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Test accuracy of model for {total} images: {round(100 * correct / total,3)}%"
    )


def save_model(model, train_data, optimizer, save_dir, epochs):
    # Saving: feature weights, new classifier, index-to-class mapping, optimiser state, and No. of epochs
    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': train_data.class_to_idx,
        'opt_state': optimizer.state_dict,
        'num_epochs': epochs
    }

    return torch.save(checkpoint, save_dir)


def load_checkpoint(model, save_dir, gpu_mode):
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(
            save_dir, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    loaded_model = load_checkpoint(model).cpu()

    image = process_image(image_path)

    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)

    image_add_dim = image_tensor.unsqueeze_(0)

    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model.forward(image_add_dim)

    probabilities = torch.exp(output)
    probabilitiesTop = probabilities.topk(topk)[0]
    indexesTop = probabilities.topk(topk)[1]

    probabilitiesTopList = np.array(probabilitiesTop)[0]
    indexTopList = np.array(indexesTop[0])

    classToIndex = loaded_model.class_to_idx

    indexToClass = {x: y for y, x in classToIndex.items()}

    classesTopList = []
    for index in indexTopList:
        classesTopList += [indexToClass[index]]

    return probabilitiesTopList, classesTopList