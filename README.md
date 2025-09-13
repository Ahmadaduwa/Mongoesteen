# Mangosteen Volume Prediction Project

A machine learning project for predicting mangosteen volume using computer vision and deep learning techniques. This project combines object detection, image processing, and regression modeling to estimate mangosteen volumes from multi-view images.

## 🎯 Project Overview

This project aims to develop an automated system for predicting mangosteen volume using:
- **Object Detection**: Roboflow-based detection of mangosteen fruits and reference markers
- **Multi-view Image Processing**: Combining images from different angles (up, down, side views)
- **Deep Learning**: Xception-based CNN for volume regression
- **Data Augmentation**: Advanced augmentation techniques for improved model performance

## 📁 Project Structure

```
Mongoesteen/
├── DataMangosteen/           # Raw dataset organized by groups
│   ├── Group 1-6/           # Different data groups
│   └── ...
├── Datasets/                # Processed datasets
│   ├── combined/            # Combined multi-view images
│   ├── labels/              # Volume labels (weight in grams)
│   ├── Image Data up/       # Top view images
│   ├── Image Data down/     # Bottom view images
│   ├── Image Data side/     # Side view images
│   └── sheet-data.xlsx      # Original volume data
├── DataTrain/               # Training dataset
├── DataTest/                # Test dataset
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs and visualizations
├── ModelTraining.py         # Main training script
├── PrepareDatasetForRegression.py  # Data preparation
├── combined.py              # Image combination script
├── crop.py                  # Image cropping script
├── label.py                 # Label generation script
├── changeName.py            # File renaming utility
└── *.ipynb                  # Jupyter notebooks for training/testing
```

## 🚀 Features

### 1. Object Detection & Cropping
- **Roboflow Integration**: Uses pre-trained model for mangosteen and marker detection
- **Automatic Cropping**: Crops mangosteen regions based on detection results
- **Reference Markers**: Uses 30mm markers for scale calibration

### 2. Multi-view Image Processing
- **Image Combination**: Combines up, down, and side view images into 224x224 composite images
- **Letterbox Padding**: Maintains aspect ratio while resizing to target dimensions
- **Data Organization**: Systematic organization of multi-view datasets

### 3. Deep Learning Model
- **Architecture**: Xception-based CNN with custom regression head
- **Transfer Learning**: Pre-trained ImageNet weights with fine-tuning
- **Data Augmentation**: Brightness, contrast, flip, rotation, and scaling augmentations
- **Regularization**: L2 regularization and dropout for overfitting prevention

### 4. Training Pipeline
- **Two-stage Training**: Initial head training followed by backbone fine-tuning
- **Callbacks**: Early stopping and learning rate reduction
- **Validation**: 20% holdout validation set
- **Scalability**: StandardScaler for volume normalization

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install joblib
pip install roboflow
```

### Roboflow Setup
1. Install Roboflow: `pip install roboflow`
2. Get API key from Roboflow dashboard
3. Update API key in scripts:
   - `crop.py`
   - `PrepareDatasetForRegression.py`

## 📊 Usage

### 1. Data Preparation

#### Crop Images
```bash
python crop.py
```
- Selects input folder containing raw images
- Detects mangosteen and markers using Roboflow
- Crops mangosteen regions to CROPPED subfolder

#### Combine Multi-view Images
```bash
python combined.py
```
- Combines up, down, and side view images
- Creates 224x224 composite images
- Outputs to `Datasets/combined2/`

#### Generate Labels
```bash
python label.py
```
- Reads volume data from Excel file
- Creates individual label files for each image
- Outputs to `Datasets/labels/`

### 2. Model Training

#### Using Python Script
```bash
python ModelTraining.py
```

#### Using Jupyter Notebooks
```bash
jupyter notebook own_train.ipynb
```

### 3. Model Testing
```bash
jupyter notebook own_test.ipynb
```

## 🔧 Configuration

### Model Parameters
```python
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 200
FINE_TUNE_EPOCHS = 0
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-6
```

### Data Augmentation
- **Brightness**: Random adjustment ±20%
- **Contrast**: Random adjustment 0.8-1.2x
- **Flip**: Random horizontal flip
- **Rotation**: Random rotation ±15°
- **Scale**: Random scaling 0.9-1.1x

## 📈 Model Performance

The model uses Mean Absolute Error (MAE) and Mean Squared Error (MSE) as evaluation metrics:

- **Training**: Monitored with validation split
- **Testing**: Evaluated on unseen test dataset
- **Scalability**: Volume values normalized using StandardScaler

## 📋 Dataset Information

### Data Groups
- **Group 1-6**: Different mangosteen datasets
- **Multi-view**: Up, down, and side view images
- **Volume Range**: Various weight measurements in grams
- **Image Format**: JPG images, 224x224 resolution

### Data Augmentation
- **Training Set**: Augmented with brightness, contrast, flip, rotation, scale
- **Validation Set**: No augmentation applied
- **Test Set**: No augmentation applied

## 🔍 Key Scripts

### `ModelTraining.py`
Main training script with:
- Xception-based architecture
- Two-stage training (head + fine-tuning)
- Data augmentation pipeline
- Callback management

### `PrepareDatasetForRegression.py`
Data preparation script with:
- Roboflow object detection
- Multi-view image processing
- Volume estimation from markers
- Dataset organization

### `combined.py`
Image combination script:
- Multi-view image merging
- Letterbox padding
- Systematic file organization

### `crop.py`
Image cropping utility:
- Object detection-based cropping
- Marker-based scale calibration
- Batch processing

## 📝 Notes

- **Roboflow Model**: Uses `mangosteen-markersc-detection-c7wdu/1`
- **Reference Markers**: 30mm square/circle markers for scale
- **Image Processing**: OpenCV-based image manipulation
- **Model Saving**: HDF5 format with scaler preservation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for research and educational purposes. Please ensure proper attribution when using this code.

## 🆘 Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Last Updated**: January 2025
**Version**: 1.0.0
