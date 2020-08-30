import json
import argparse
import torch
import PIL
from torchvision import datasets, transforms, models
import numpy as np
from torch import nn

def get_args():
    parser = argparse.ArgumentParser(
        description = 'Prediction Arguments for the flexible image classifier script'
    )
    
    parser.add_argument('input', action = 'store')
    parser.add_argument('checkpoint', action = 'store')
    parser.add_argument('--top_k', action = 'store', dest = 'top_k', default = 5, type = int)
    parser.add_argument('--category_names', action = 'store', dest = 'category_names', default = 'cat_to_name.json')
    parser.add_argument('--gpu', action = 'store_true', default = False, dest = 'gpu')
    
    results = parser.parse_args()
#     print("input:           {}".format(results.input))
#     print("checkpoint:      {}".format(results.checkpoint))
#     print("top_k:           {}".format(results.top_k))
#     print("category_names:  {}".format(results.category_names))
#     print("gpu:             {}".format(results.gpu))

    return results
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    
    model_vgg13 = models.vgg13(pretrained=True)
    model_densenet121 = models.densenet121(pretrained=True)
    
    aval_models = {'vgg13': model_vgg13, 'densenet121': model_densenet121}
    model = aval_models[arch]
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.classifier = checkpoint['classifier']  # NOTE: we need to checkpoint our classfier as well, otherwise an error of mismatch classifier will occur
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    test_img = PIL.Image.open(image)
    
    # Get original dimensions
    orig_width, orig_height = test_img.size
    orig_scale = orig_height / orig_width
    
    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: 
        resize_size=(256, round(256*orig_scale))
    else: 
        resize_size=(round(256/orig_scale), 256)
    
    resized_test_img = test_img.resize(resize_size)
    

    # Find pixels to crop on to create 224x224 image
    crop_size = 224
    size = resized_test_img.size
    
    left = (size[0] - crop_size)/2
    top = (size[1] - crop_size)/2
    right = (left + crop_size)
    bottom = (top + crop_size)
    
    cropped_test_img = resized_test_img.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_img = np.array(cropped_test_img)/255 

    # Normalize each color channel
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    np_img = (np_img - means) / stds
        
    # Set the color to the first channel
    np_img = np_img.transpose(2, 0, 1)
    
    return torch.from_numpy(np_img)
   
    
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    args = get_args()
    category_names = args.category_names
    gpu = args.gpu
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print("--- currently using {} ---".format(device))
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)

    model.eval()
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
    
    # Find probabilities (results) by passing through the forward function 
    # (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)
    
    # Convert to linear scale
    linear_probs = torch.exp(log_probs)
    
    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(topk)
    
    top_probs = top_probs.detach().cpu().numpy()[0]
    top_cat_idxes = top_labels.detach().cpu().numpy()[0]
    
    idx_to_class = {idx: cat for cat, idx in model.class_to_idx.items()}
    
    top_cat_codes = [idx_to_class[idx] for idx in top_cat_idxes]
    top_flower_names = [cat_to_name[code] for code in top_cat_codes]
    
    return top_probs, top_cat_codes, top_flower_names


def eval_test_data_accuracy(model):
    test_loss = 0
    accuracy = 0
    model.eval()
    criterion = nn.NLLLoss()
    
    test_dir = 'flowers/test'
    test_transforms = transforms.Compose([
                                       transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                    ])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    args = get_args()
    gpu = args.gpu
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
    #         print(inputs.shape)
    #         print(type(inputs))
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

    model.train()


def main():
    args = get_args()
    checkpoint = args.checkpoint
    image = args.input
    top_k = args.top_k
    
    model = load_checkpoint(checkpoint)
    img_torch_array = process_image(image)
    top_probs, top_cat_codes, top_flower_names = predict(image, model, top_k)
    
    print(top_probs)
    print(top_flower_names)
  
    eval_test_data_accuracy(model)
    
    
if __name__ == "__main__":
    main()


