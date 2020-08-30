import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
import numpy as np
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser(
                description = 'Training Arguments for the flexible image classifier script'
            )

    parser.add_argument('data_dir', action = 'store')
    parser.add_argument('--save_dir', action = 'store', dest = 'save_dir', default = '')
    parser.add_argument('--arch', action = 'store', dest = 'arch', default = 'densenet121')
    parser.add_argument('--learning_rate', action = 'store', dest = 'learning_rate', default = 0.003, type = float)
    parser.add_argument('--hidden_units', action = 'store', dest = 'hidden_units', default = 4096, type = int)
    parser.add_argument('--epochs', action = 'store', dest = 'epochs', default = 1, type = int)
    parser.add_argument('--gpu', action = 'store_true', default = False, dest = 'gpu')

    results = parser.parse_args()
#     print('data_dir     = {!r}'.format(results.data_dir)
#     print('save_dir     = {!r}'.format(results.save_dir)
#     print('arch         = {!r}'.format(results.arch))
#     print('learning_rate= {!r}'.format(results.learning_rate))
#     print('hidden_units = {!r}'.format(results.hidden_units))
#     print('epochs       = {!r}'.format(resul.epochs))
#     print('gpu          = {!r}'.format(results.gpu))

    return results
          
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                ])

    valid_transforms = transforms.Compose([
                                       transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                    ])

    test_transforms = transforms.Compose([
                                       transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                    ])
    
    data_transforms = {
        'train': train_transforms,
        'valid': valid_transforms,
        'test': test_transforms
    }

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    image_datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }

    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)

    dataloaders = {
        'train': trainloader,
        'valid': validloader,
        'test': testloader
    }
    
    return dataloaders, image_datasets
    
def save_model(model, image_datasets, args):
    save_dir = args.save_dir
    full_save_path = save_dir + 'flexible_classifier_checkpoint.pth'

    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'arch': args.arch
    }

    torch.save(checkpoint, full_save_path)
    print('--- model saved ---')
    
def train(dataloaders, args):
    gpu = args.gpu
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print("--- currently using {} ---".format(device))
    
    model_vgg13 = models.vgg13(pretrained=True)
    model_densenet121 = models.densenet121(pretrained=True)
    
    aval_models = {'vgg13': model_vgg13, 'densenet121': model_densenet121}
    models_input_size = {'vgg13': 25088, 'densenet121': 1024}
    
    arch = args.arch
    model = aval_models[arch]
    input_size = models_input_size[arch]
    hidden_units = args.hidden_units
    output_size = 102
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    trainloader = dataloaders['train']
    validloader = dataloaders['valid']

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    return model
    
    
def main():
    args = get_args()
    data_dir = args.data_dir
    # save_dir = args.save_dir
    # arch = args.arch
    # learning_rate = args.learning_rate
    # hidden_units = args.hidden_units
    # epochs = args.epochs
    # gpu = args.gpu

    dataloaders, image_datasets = load_data(data_dir)
    model = train(dataloaders, args)
    save_model(model, image_datasets, args) 
    
    
if __name__ == "__main__":
    main()
   


