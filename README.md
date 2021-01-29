# GTSRB_solution
 Explore the "Germany Traffic Sign Recognition Benchmark" dataset, implement some method(s) for traffic sign classification

# Data Processing

The training data and test data already have been reallocated to 43 different folders based on the image classes. The ImageFolder function of pytorch can read these image data and generate 3-channel feature matrix with labels.

# Environment Requirement and Execution<br>
Python3
Dependent Package: pytorch, numpy, sklearn, skimage, ThunderSVM(https://github.com/Xtra-Computing/thundersvm)<br>
Execution:<br>
Use command: <br>
python3 SVM.py<br>
python3 basic_cnn.py<br>
python3 improved.py<br>

# Method

1. HOG+SVM <br>
I used skimage and scikit-learn for HOG and SVM. For the multi-class, single-image classification task, I use one- versus-one (OVO) model, which require k(k-1)/2 SVM classifiers. I used ThunderSVM [2], a GPU-enable framework, which significantly improved the training speed of the model.<br>
**Train accuracy: 78.32%**<br>
**Test accuracy: 76.79%**<br>

2. Basic CNN(2-layer)<br>
I used convolutional neural network for the multi-class image classification. First of all, I have used a very basic 2-layer CNN for the image classification task. The architecture of the basic 2-layer CNN is shown below:<br>
![image](https://github.com/JiarongFan/GTSRB_solution/blob/master/basic_cnn_arch.png)<br>
**Train accuracy: 95%**<br>
**Test accuracy: 86%**<br>
The detail of each class accuracy see the technical report and cm.txt (confusion matrix)<br>

3. Improved CNN<br>
To achieve higher accuracy of the CNN classifier, I have modified the basic 2-layer CNN architecture. I have added more convolutional layer and deeper neural network. In order to avoid overfitting, I have added dropout mechanism. I also pick up some ideas from VGG16, the convolutional layer only changes the deep of the image and the pooling layer only change the width and height of the image. The improved 6-layer CNN is shown below:<br>
![image](https://github.com/JiarongFan/GTSRB_solution/blob/master/improve_cnn_arch.png)<br>
**Train accuracy: 97%**<br>
**Test accuracy: 91%**<br>
The detail of each class accuracy see the technical report and cm.txt (confusion matrix)<br>

#optional task: GANs for image generation
Generative models can be used to generate similar or different style images from a dataset, which can be used in data augmentation. Ian Goodfellow proposed Generative Adversarial Network (GNN), which can generate similar or different style images by the game of two neural networks. I modified a demo so that it can work on the GTSRB dataset<br>
Execution:<br>
Use command: <br>
python3 gans.py<br>
The result of GAN:<br>
Real image:<br>
![image](https://github.com/JiarongFan/GTSRB_solution/blob/master/real_samples.png)<br>
Fake image by GANs:<br>
![image](https://github.com/JiarongFan/GTSRB_solution/blob/master/fake_samples_epoch_024.png)<br>


