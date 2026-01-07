# Skin Care Product Recommendation System

A deep learning-based multi-label classification system for detecting skin problems from images using PyTorch. The system leverages pretrained CNN models (EfficientNet-B0, ResNet50, ViT) to identify 21 different skin conditions, which can be used to recommend appropriate skincare products.

## 🎯 Features

- **Multi-Label Classification**: Detect multiple skin conditions simultaneously from a single image
- **21 Skin Condition Classes**: Including acne, dry skin, oily skin, wrinkles, dark spots, and more
- **Multiple Model Architectures**: Support for EfficientNet-B0, ResNet50, and Vision Transformer (ViT)
- **Transfer Learning**: Utilizes pretrained weights for improved accuracy
- **Easy Prediction API**: Simple functions for single image and batch predictions

## 📋 Supported Skin Conditions

The model can detect the following 21 skin conditions:

| Condition | Condition | Condition |
|-----------|-----------|-----------|
| Enlarged Pores | Acne | Acne Marks |
| Acne Scar | Blackhead | Burned Skin |
| Dark Circle | Dark Spot | Dry |
| Freckle | Melasma | Nodules |
| Normal Skin | Oily | Papules |
| Pores | Pustules | Skin Redness |
| Vascular | Whitehead | Wrinkle |

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/anish-bk/Skin-Care-Product-Recommendation-System.git
cd Skin-Care-Product-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
├── config.py          # Configuration parameters and argument parsing
├── dataset.py         # Dataset class and data loading utilities
├── model.py           # Neural network architecture definition
├── train.py           # Training and validation functions
├── evaluate.py        # Evaluation metrics and testing
├── predict.py         # Inference utilities for new images
├── main.py            # Main entry point for training pipeline
└── requirements.txt   # Python dependencies
```

## 🚀 Usage

### Training

Train the model with default configuration:
```bash
python main.py --data-dir /path/to/dataset
```

Train with custom parameters:
```bash
python main.py --data-dir ./Skin-Problem-Detection-Multiple-Dataset \
               --model efficientnet_b0 \
               --batch-size 32 \
               --epochs 20 \
               --lr 0.0001 \
               --image-size 224 \
               --save-path best_model.pth
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `Skin-Problem-Detection-Multiple-Dataset` | Path to dataset directory |
| `--model` | `efficientnet_b0` | Model architecture (efficientnet_b0, resnet50, vit_base_patch16_224) |
| `--batch-size` | `32` | Training batch size |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `0.0001` | Learning rate |
| `--image-size` | `224` | Input image size |
| `--save-path` | `best_skin_classifier.pth` | Path to save the best model |

### Prediction

#### Single Image Prediction

```python
from predict import predict_single_image

# Predict skin conditions for a single image
results = predict_single_image("path/to/skin_image.jpg", threshold=0.5)

for condition, data in results.items():
    if data['prediction'] == 1:
        print(f"{condition}: {data['probability']:.2%}")
```

#### Batch Prediction

```python
from predict import predict_batch

# Predict for multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = predict_batch(image_paths, threshold=0.5)

for result in results:
    print(f"Image: {result['image_path']}")
    # Process predictions...
```

## 🏗️ Model Architecture

The system uses a **MultiLabelSkinClassifier** with the following architecture:

1. **Backbone**: Pretrained model (EfficientNet-B0/ResNet50/ViT) for feature extraction
2. **Dropout Layer**: 30% dropout for regularization
3. **Classification Head**: Linear layer mapping features to 21 output classes

### Key Design Choices

- **BCEWithLogitsLoss**: Combines sigmoid activation with binary cross-entropy for numerical stability
- **No mutual exclusivity**: Each skin condition is predicted independently (multi-label, not multi-class)
- **Transfer Learning**: Pretrained ImageNet weights provide robust feature extraction

## 📊 Dataset Structure

The expected dataset structure:
```
Skin-Problem-Detection-Multiple-Dataset/
├── train/
│   ├── images/
│   └── labels.csv
├── valid/
│   ├── images/
│   └── labels.csv
└── test/
    ├── images/
    └── labels.csv
```

## 📦 Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (PyTorch Image Models)
- pandas >= 2.0.0
- numpy >= 1.24.0
- Pillow >= 9.5.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0

## 🔧 Configuration

All configuration parameters can be found in `config.py`. Key settings include:

- **Model Configuration**: Architecture selection, number of classes, pretrained weights
- **Training Hyperparameters**: Batch size, epochs, learning rate, weight decay
- **Image Preprocessing**: Image size, normalization parameters
- **Device Configuration**: Automatic GPU detection

## 📈 Training Pipeline

The training pipeline in `main.py` follows these steps:

1. **Configuration**: Parse arguments and setup configuration
2. **Data Loading**: Create train/validation/test data loaders
3. **Model Creation**: Initialize pretrained model with custom head
4. **Training Loop**: Train with validation and save best model
5. **Evaluation**: Final evaluation on test set with detailed metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

- **Anish Bishwakarma** - [GitHub](https://github.com/anish-bk)

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained models
- The open-source community for various skin condition datasets
