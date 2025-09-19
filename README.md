#Image Classification using PyTorch
A deep learning project that classifies animal faces into three categories (cat, dog, wild) using transfer learning with PyTorch. This implementation demonstrates a complete end-to-end computer vision pipeline achieving exceptional accuracy with minimal overfitting.


📊 Performance Metrics
Training Accuracy: 99%+
Validation Accuracy: 98.5%+
Test Accuracy: 99.2%+
Precision, Recall, F1-score: > 0.99 for all classes
Overfitting: Minimal (nearly identical training/validation loss curves)


🚀 Features
Transfer learning with pre-trained ResNet50
Comprehensive data preprocessing and augmentation
Custom PyTorch Dataset implementation
Advanced training with learning rate scheduling
Detailed model evaluation and visualization
Real-time inference capability
Confusion matrix and classification reports


📁 Dataset
The project uses the Animal Faces HQ (AFHQ) dataset from Kaggle:
Source: https://www.kaggle.com/datasets/andrewmvd/animal-faces
Classes: Cat, Dog, Wild animals
Total Images: 16,130
Split: 70% training, 15% validation, 15% testing



🏗️ Model Architecture
Base Model: ResNet50 (pre-trained on ImageNet)
Custom Head:
Dropout (0.5)
Linear layer (2048 → 512 units)
ReLU activation
Dropout (0.3)
Linear layer (512 → 3 units)


📈 Results
The model achieves exceptional performance across all metrics:
Overall Accuracy: 99.2%
Class-wise Performance:
Cat: Precision=0.99, Recall=0.99, F1-score=0.99
Dog: Precision=0.99, Recall=0.99, F1-score=0.99
Wild: Precision=0.99, Recall=0.99, F1-score=0.99



🧠 Key Techniques
Transfer Learning: Leveraged pre-trained ResNet50 weights
Data Augmentation: Comprehensive transformation pipeline
Regularization: Dropout layers to prevent overfitting
Learning Rate Scheduling: Step decay for better convergence
Early Stopping: Model checkpointing based on validation accuracy
