import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载归一化后的数据集
data_dir = '/home/phd/datasets/breakHis/'
train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #batch_size=16, batch_size=8, batch_size=64
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义预训练模型
models_dict = {
    'ResNet': models.resnet18(pretrained=True), #'ResNet': models.resnet18(pretrained=False)
    'DenseNet': models.densenet121(pretrained=True),
    'Inception': models.inception_v3(pretrained=True),
    'EfficientNet': models.efficientnet_b0(pretrained=True)
}

# 修改模型的最后一层以适应数据集的类别数
num_classes = len(train_dataset.classes)
for name, model in models_dict.items():
    if name == 'ResNet':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'DenseNet':
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'Inception':
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'EfficientNet':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# 训练和评估函数
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if isinstance(model, torchvision.models.Inception3):
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 训练和评估所有模型
accuracies = {}
#epochs = 10
#epochs = 20
#epochs = 30
epochs = 50
for name, model in models_dict.items():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    accuracy = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs)
    accuracies[name] = accuracy
    print(f'{name} Accuracy: {accuracy}')

# 绘制准确率比较图
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Pretrained Models')
plt.show()

# 生成热力图（以 ResNet 为例）
target_layer = models_dict['ResNet'].layer4[-1]
cam = GradCAM(model=models_dict['ResNet'], target_layer=target_layer)

# 选择一张测试图像
test_image, test_label = test_dataset[0]
test_image = test_image.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 生成热力图
grayscale_cam = cam(input_tensor=test_image)
grayscale_cam = grayscale_cam[0, :]
rgb_img = test_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.title('GradCAM Heatmap for ResNet')
plt.show()

    