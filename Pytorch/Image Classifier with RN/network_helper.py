# Imports here
import os
import numpy as np
import torch

import torchvision
from collections import OrderedDict
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.optim as optim
import matplotlib.pyplot as plt


train_on_gpu = torch.cuda.is_available()
   
'''
Data_dir = Diretorio onde está as imagens
Batch_size é o numero de elementos que será carregado a cada iteração, serve para não carregar todos os dados de uma unica vez e deixar o treinamento do modelo muito lerdo.
Image_size = Tamanho da imagem
Norm_mean = É o normalize the means
Norm_std = É o standard deviations
Os valores da normalização da media e do desvio padrão vai modificar a cor das imagens para ficar centralizado em 0 com range entre -1 e 1.
'''
def load_dataset(data_dir='flower_data', batch_size=20, image_size=224, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    #Para carregar é necessário que já o dataset encontre-se no mesmo diretorio que esse arquivo.
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    data_transforms = transforms.Compose([transforms.RandomResizedCrop(image_size),transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])

    #É necessário carregar os dados de teste e de treino, e realizar uma transformação dos dados para deixar todos no mesmo padrão.
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

   
    num_workers=0

    # Prepara os DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                              num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader

def create_network(numb_output):
    ##Alterar a quantidade de ouput conforme o dataset escolhido, nesse caso 102
    from collections import OrderedDict
    import torch.nn as nn
    
    redeNeural = models.resnet18(pretrained=True)
    n_inputs = redeNeural.fc.in_features   
    last_layer = nn.Linear(n_inputs, numb_output)
    redeNeural.fc = last_layer
    return redeNeural
    
def _create_grafico_accuracy(train_losses, test_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    
def _validation(model, testloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    accuracy = 0
    for batch_i, (images, labels) in enumerate(testloader):
        
        if train_on_gpu:
            images, labels = images.cuda(), labels.cuda()

        output = model(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy
    

def train_network(redeNeural, train_loader, numb_epochs=20, name_file='model-saved.pth'):
    #Antes de validar o modelo é preciso incluir o tipo de criterion e optimizer
    
    if train_on_gpu:
        redeNeural.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(redeNeural.fc.parameters(), lr=0.001)
    
    # Epoch é algo semelhante a iteração no caso de ML(Machine Learning). 
    # Quanto mais Epoch mais o modelo será treinado e terá um desempenho melhor. Mas é necessário tomar cuidado com Overfitting.
    globalLoss = 100;
    for epoch in range(1, numb_epochs+1):

        # A perda de treino será incrementada a cada batch 
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for batch_i, (data, target) in enumerate(train_loader):

            ##Utilizar GPU caso esteja disponivel.
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = redeNeural(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss 
            train_loss += loss.item()
            if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
                train_loss_mean = train_loss / 20;
                print('Epoch %d, Batch %d loss: %.16f' % (epoch, batch_i + 1, train_loss_mean))
                if  train_loss_mean <= globalLoss:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    globalLoss,
                    train_loss_mean))
                    torch.save(redeNeural.state_dict(), name_file)
                    globalLoss = train_loss_mean
                train_loss = 0.0

                
def _reload_module(file='model-saved.pth'):
    return torch.load(file)           
                
def valid_network(test_loader, redeNeural, classes, criterion = nn.CrossEntropyLoss()):
    test_loss = 0.0
    size = len(classes);

    if train_on_gpu:
        redeNeural.cuda()
        
    class_correct = list(0. for i in range(size))
    class_total = list(0. for i in range(size))

    redeNeural.eval() # prep model for evaluation
    test_losses = []

    for batch_i, (data, target) in enumerate(test_loader):
        # forward pass: compute predicted outputs by passing inputs to the model
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = redeNeural(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(pred.size()[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    redeNeural.train() # return model to train
    
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    test_losses.append(test_loss/len(test_loader))
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    
def show_image_validate(test_loader, redeNeural, classes):
    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # move model inputs to cuda, if GPU available
    if train_on_gpu:
        imagesCuda = images.cuda()
        redeNeural.cuda()
        
    output = redeNeural(imagesCuda)

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

    images = images.numpy()

    # plot the images in the batch, along with predicted and true labels
    count = 0
    fig = plt.figure(figsize=(37, 8))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)).astype(np.uint8))
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                     color=("green" if preds[idx]==labels[idx].item() else "red"))
        if preds[idx].item()==labels[idx].item():
            count +=1
    print("Acertos: ", count)
