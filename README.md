# MagNet
 
## 1. Introduction
MagNet is a deep learning - based model designed for specific image - related tasks, potentially including image segmentation and classification. It consists of an encoder - decoder architecture with additional components like the SDSB decoder level and a post - refinement block (PRB). The model also incorporates stain normalization techniques to handle variations in staining in biological images.

## 2 Dataset Structure
The BreakHis dataset contains 7,936 digitized histopathological images, which are split into two main categories: benign and malignant. Each image has a histological diagnosis, the patient's age, and other clinical details. The images are further classified into sub - categories such as ductal carcinoma, lobular carcinoma, etc. and are captured at four distinct magnifications: 40x, 100x, 200x, and 400x.

## 3. Stain Normalization Technique
Before applying the MagNet model, the images in the dataset are pre - processed using a stain normalization technique. In this study, the Adaptive Color Deconvolution (ACD) technique is employed for stain normalization.
- **Explanation**: ACD is used to isolate the stain features via a unified adjustment after extracting the individual stain components from each pixel of the breast histopathology image. The process involves transforming the RGB values of the image to Optical Density (OD) values, creating the Stain Color Appearance (SCA) matrix, and calculating the stain densities. These stain densities are then optimized using an objective function and the gradient descent optimization algorithm to achieve effective stain normalization.
- **Implementation Details**: The code for stain normalization can be found in [stain_normalization.py] (if available in the repository). It contains functions to perform the necessary calculations as described in the paper.

## 4. Environment Setting
### 4.1 Prerequisites
- **Operating System**: This code has been tested on Linux Ubuntu 20.04.
- **Python Version**: Python 3.7 
- **Python Libraries**:
    - `torch`: For deep learning model implementation.
    - `torchvision`: For handling image data and pre - trained models.
    - `numpy`: For numerical operations..
    - Other relevant libraries such as `pytorch_grad_cam`, `matplotlib` (for visualization), etc.
    - 
## 5. Repository Structure
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

## 6. Installation
### 6.1 Clone the Repository
```bash
git clone https://github.com/SaeedIqbal/MagNet.git
cd MagNet
```

### 6.2 Install Dependencies
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

## 7. Usage

### 7.1 Stain Normalization
If you want to normalize the staining of a dataset, you can use the following code:
```python
from MagNet.stain_normalization import stain_normalize_dataset

data_dir = "your_dataset_directory"
output_dir = "normalized_dataset"
target_image_path = "target_image.jpg"

stain_normalize_dataset(data_dir, output_dir, target_image_path)
```

### 7.2 Loading the Dataset
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

### 7.3 Training the MagNet Model
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

### 7.4 Training and Evaluating Pre - trained Models
You can also train and evaluate multiple pre - trained models using the `pretrained_models_training.py` script:
```python
python pretrained_models_training.py
```

## 8. Model Architecture
### 8.1 Encoder
The encoder consists of multiple `PUNetEncoderUnit` blocks. Each `PUNetEncoderUnit` uses an `SDC` block followed by a smoothed convolution to downsample the feature maps.

### 8.2 Decoder
The decoder consists of transposed convolution and convolution layers to upsample the feature maps.

### 8.3 SDSB Decoder Level
The `SDSBDecoderLevel` upsamples the encoder features, fuses them, and then applies depth and spatial scaling convolutions.

### 8.4 Post - Refinement Block (PRB)
The `PRB` upsamples the decoder features, fuses them, and then applies a final convolution with a sigmoid activation to get the final output.

## 9. Citation
If you use the code or results from this repository in your research, please cite the original paper:
```bibtex
@article{IQBAL2024108222,
title = {Adaptive magnification network for precise tumor analysis in histopathological images},
journal = {Computers in Human Behavior},
volume = {156},
pages = {108222},
year = {2024},
issn = {0747-5632},
doi = {https://doi.org/10.1016/j.chb.2024.108222},
url = {https://www.sciencedirect.com/science/article/pii/S0747563224000906},
author = {Saeed Iqbal and Adnan N. Qureshi and Khursheed Aurangzeb and Musaed Alhussein and Muhammad Shahid Anwar and Yudong Zhang and Ikram Syed},
keywords = {Breast cancer histopathology, Magnification invariance, Stain normalization, Multi-level features, Cancer diagnostics, Histopathological images},
abstract = {The variable magnification levels in histopathology images make it difficult to accurately categorize tumor regions in breast cancer histology. In this study, a novel architecture for accurate image interpretation MagNet is presented. With specific modules like Separable Dilation Convolution (SDC), Separable Dilation Skip Block (SDSB), and Point-wise Reformation Block (PRB), MagNet uses a Parallel U-Net (PU-Net) infrastructure. SDC in the PU-Net encoder ensures downsampled generalized feature representations by capturing characteristic attributes at varying magnifications. Using feature upsampling, attribute mapping merging, and PRB for precise feature capture, the decoder improves reconstruction. While PRB combines data from several decoder levels, SDSB creates vital links between the encoder and decoder layers. MagNet requires less processing of histopathology images and improves multi-magnification feature maps. MagNet performs exceptionally well, constantly outperforming rivals in terms of accuracy (0.98), F1 score (0.97), sensitivity (0.96), and specificity (0.97). The effectiveness of MagNet and its revolutionary potential in cancer diagnostics are shown by these quantitative data.}
}
```

## 10. Contact
If you have any questions or issues regarding the code, dataset, or reproduction of the experiments, please contact the corresponding author of the paper [author's email address] or create an issue in the GitHub repository. 
