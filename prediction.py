"""This script is used to test the accuracy of the ResNet50 model on the MNIST test dataset."""

import torch
from nets.resnet50 import ResNet,Bottleneck
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import cv2
import time
from tqdm import tqdm

# Load model
# Path to the pretrained model
PATH = './logs/resnet50-mnist.pth'
# Ask user for batch size
# Batch_Size = int(input('The number of handwritten font images predicted each times：'))
Batch_Size = 1
# model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
model = ResNet(Bottleneck, [3, 4, 6, 3],custom_conv_layer_index=1)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model = model.cpu()
model.eval()

#Load test dataset
test_dataset = datasets.MNIST(root='data/', train=False,
                                    transform=transforms.ToTensor(), download=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=Batch_Size, shuffle=False)

# Accuracy evaluation
correct = 0
total = 0
# with torch.no_grad():
#     pbar = tqdm(test_loader, total=len(test_loader), desc="Testing")
#     for images, labels in pbar:
#         # print(f"\nBatch {i}: loaded")
#         images = images.cpu()
#         labels = labels.cpu()
        
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
        
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         accuracy = 100 * correct / total
#         pbar.set_postfix({'Accuracy (%)': f"{accuracy:.2f}"})

# 🚨 after this: TEST_CODE

n_dat = 1
with torch.no_grad():
    pbar = tqdm(test_loader, total=n_dat, desc="Testing")
    idx = 0
    for images, labels in pbar:
        if (idx == n_dat):
            break
        # print(images.shape, labels.shape)
        images = images.cpu()
        labels = labels.cpu()
        # print("predicting...")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        idx += 1
        total += labels.size(0)
        # print("predicted: ", predicted, " / labels: ", labels)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        pbar.set_postfix({'Accuracy (%)': f"{accuracy:.2f}"})

accuracy = 100 * correct / total
print(f"✅ Accuracy on the MNIST test set: {accuracy:.2f}%")

# Demo prediction visualization
# while True:

#     images, labels = next(iter(test_loader))
#     img = torchvision.utils.make_grid(images, nrow=Batch_Size)
#     img_array = img.numpy().transpose(1, 2, 0)

#     start_time = time.time()
#     outputs = model(images.cuda())
#     _, predicted = torch.max(outputs.data, 1)
#     end_time = time.time()

#     print('Estimated time:', end_time - start_time)
#     print('Predicted:', predicted.cpu().numpy())
#     print('Ground Truth:', labels.cpu().numpy())
        
#     cv2.imshow('img', img_array)
#     cv2.waitKey(0)


