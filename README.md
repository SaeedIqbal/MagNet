# MagNet
 
## 1. Introduction
MagNet is a deep learning - based model designed for specific image - related tasks, potentially including image segmentation and classification. It consists of an encoder - decoder architecture with additional components like the SDSB decoder level and a post - refinement block (PRB). The model also incorporates stain normalization techniques to handle variations in staining in biological images.

## 2. Repository Structure
- `MagNet/`:
  - `SDC.py`: Defines the Spatial Dilated Convolution (SDC) block used in the encoder.
  - `PUNetEncoderUnit.py`: Contains the PUNet encoder unit which uses the SDC block.
  - `sdsb_decoder.py`: Implements the SDSB decoder level.
  - `PRB.py`: Defines the Post - Refinement Block.
  - `MagNet.py`: The main model file that combines the encoder, decoder, SDSB decoder, and PRB.
  - `stain_normalization.py`: Provides functions for stain normalization of images and datasets.
  - `loadDataset.py`: Helps in loading the BreakHis dataset.
  - `pretrained_models_training.py`: Trains and evaluates multiple pre - trained models.
  - `main.py`: The main script to train the MagNet model.
- `README.md`: Documentation for the project.

## 3. Installation
### 3.1 Clone the Repository
```bash
git clone https://github.com/SaeedIqbal/MagNet.git
cd MagNet
```

### 3.2 Install Dependencies
The project requires the following libraries:
- PyTorch
- NumPy
- OpenCV
- tqdm
- torchvision
- scikit - learn
- pytorch_grad_cam

You can install them using `pip`:
```bash
pip install torch numpy opencv - python tqdm torchvision scikit - learn pytorch_grad_cam
```

## 4. Usage

### 4.1 Stain Normalization
If you want to normalize the staining of a dataset, you can use the following code:
```python
from MagNet.stain_normalization import stain_normalize_dataset

data_dir = "your_dataset_directory"
output_dir = "normalized_dataset"
target_image_path = "target_image.jpg"

stain_normalize_dataset(data_dir, output_dir, target_image_path)
```

### 4.2 Loading the Dataset
To load the BreakHis dataset, you can use the `BreakHisDataset` class:
```python
from MagNet.loadDataset import BreakHisDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data_dir = '/home/phd/datasets/breakHis/'
dataset = BreakHisDataset(data_dir, transform)
train_loader = dataset.get_loader()
```

### 4.3 Training the MagNet Model
To train the MagNet model, you can use the `main.py` script:
```python
from MagNet.MagNet import MagNet
from MagNet.loadDataset import BreakHisDataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

data_dir = '/home/phd/datasets/breakHis/'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = BreakHisDataset(data_dir, transform)
train_loader = dataset.get_loader()

model = MagNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 10

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}, Loss: {loss.item()}')

train_model(model, train_loader, criterion, optimizer, epochs)
```

### 4.4 Training and Evaluating Pre - trained Models
You can also train and evaluate multiple pre - trained models using the `pretrained_models_training.py` script:
```python
python pretrained_models_training.py
```

## 5. Model Architecture
### 5.1 Encoder
The encoder consists of multiple `PUNetEncoderUnit` blocks. Each `PUNetEncoderUnit` uses an `SDC` block followed by a smoothed convolution to downsample the feature maps.

### 5.2 Decoder
The decoder consists of transposed convolution and convolution layers to upsample the feature maps.

### 5.3 SDSB Decoder Level
The `SDSBDecoderLevel` upsamples the encoder features, fuses them, and then applies depth and spatial scaling convolutions.

### 5.4 Post - Refinement Block (PRB)
The `PRB` upsamples the decoder features, fuses them, and then applies a final convolution with a sigmoid activation to get the final output.

## 6. License
Please refer to the repository for the license information.

## 7. Contributing
If you want to contribute to this project, please fork the repository and submit a pull request. Make sure to follow the coding style and add appropriate tests.

## 8. Contact
If you have any questions or suggestions, please contact the repository owner.
