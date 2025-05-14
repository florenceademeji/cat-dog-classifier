# 🐶🐱 CNN – Cat-Dog Image Classification

This project demonstrates an end-to-end deep learning workflow aimed at classifying images as either cats or dogs using a Convolutional Neural Network (CNN). The model was further improved using transfer learning with MobileNetV2 and made interpretable with Grad-CAM visualizations.

1. Data Acquisition and Setup

The dataset used is the Cats vs Dogs dataset, originally from Kaggle and available in a preprocessed form through TensorFlow’s dataset repository.
	•	Classes: cats, dogs (binary classification)
	•	Structure: Images are divided into train and validation folders
	•	Source: TensorFlow filtered version of the Kaggle Cats vs Dogs dataset

2. Data Preprocessing

Before training, the image data undergoes essential preprocessing:
	•	Image Loading: Images are read using ImageDataGenerator.flow_from_directory()
	•	Resizing: All images are resized to 150x150 (for custom CNN) and 160x160 (for MobileNetV2)
	•	Normalization: Pixel values are scaled to range [0, 1] by dividing by 255
	•	Augmentation: Training images are augmented with:
	•	Horizontal flips
	•	Rotations
	•	Zoom and shear transformations
	•	Splitting: Dataset is divided into training and validation sets (2000 and 1000 images respectively)

3. Model Architecture

Custom CNN:

A simple, yet effective CNN was built using the Keras Sequential API:
	•	3× Conv2D + MaxPooling layers
	•	Flatten + Dense(512)
	•	Sigmoid output for binary classification
	•	ReLU activations used in hidden layers

Transfer Learning with MobileNetV2:

To improve performance and leverage pretrained features:
	•	Base Model: MobileNetV2 pretrained on ImageNet
	•	Top Layers:
	•	GlobalAveragePooling2D
	•	Dropout (0.2)
	•	Dense(1, activation=‘sigmoid’)
	•	Fine-Tuning: Top layers of MobileNetV2 were unfrozen and retrained
 
4. Model Compilation and Training
	•	Loss Function: Binary Crossentropy (since it’s a 2-class problem)
	•	Optimizer: Adam (with learning rate adjusted during fine-tuning)
	•	Metrics: Accuracy
	•	Epochs:
	•	Custom CNN: 10 epochs
	•	MobileNetV2: 5 frozen + 5 fine-tuned epochs

5. Data Augmentation

To improve generalization and reduce overfitting, the following augmentations were applied to the training set:
	•	Random horizontal flip
	•	Rotation
	•	Width/height shift
	•	Zoom and shear transformations

These augmentations were done using ImageDataGenerator in Keras.

6. Evaluation

Both models were evaluated on the validation set:
Model                         Validation Accuracy
Custom CNN                       ~85%
MobileNetV2 (fine-tuned)          ~91%
Validation accuracy and loss curves were plotted to compare performance and training behavior.

7. Grad-CAM: Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize the parts of the image the CNN was focusing on while making predictions.
	•	Highlights key regions (e.g., ears, face shape)
	•	Confirms that the model is learning relevant features
	•	Applied on both cat and dog predictions

Example Grad-CAM visualization:
(Insert image here if available)

8. Conclusion

This project demonstrates how Convolutional Neural Networks and transfer learning can be applied to binary image classification tasks. Combining data preprocessing, augmentation, a custom CNN, and fine-tuned MobileNetV2, the model achieved high validation accuracy and interpretability with Grad-CAM.

This workflow is deployable and extendable to more complex image classification problems or real-world use cases like pet recognition apps or animal monitoring systems.
