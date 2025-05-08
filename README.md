# Nudity Detection using Hybrid ResNet-50 + Swin Transformer
This project implements a nudity detection model using a hybrid approach that combines a Convolutional Neural Network (ResNet-50) with a Transformer-based model (Swin Transformer). The model is fine-tuned to classify images into three categories: regular, semi-nudity, and full-nudity.

# Features

- Uses a hybrid architecture combining ResNet-50 and Swin Transformer.
- Implements Focal Loss and Soft Focal Loss to handle class imbalance.
- Data augmentation tailored for each class.
- Mixup augmentation for improved generalization.
- Cosine Annealing LR scheduler for adaptive learning rate decay.
- Early stopping mechanism based on F1 score.
- Generates a confusion matrix for evaluation.

# Model Evaluation

After training, the best model is saved in `checkpoints/best_model.pth`. The final evaluation results include:

- Accuracy and F1 Score
- Classification Report
- Confusion Matrix (visualized)
