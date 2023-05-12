---
title: "Industrial Biscuit Anomaly Detection"
description: Leveraging deep learning for quality control in biscuit manufacturing. Utilizing Xception and EfficientNetB4 architectures, this project achieves unprecedented accuracy in detecting anomalies, ensuring higher product quality and efficiency in the production process."
pubDate: "May 5 2023"
heroImage: "/cookie.png"
---

## Dataset Description and Preprocessing

The Industrial Biscuits (Cookie) dataset is an internal dataset specifically curated for the anomaly detection task. It consists of images featuring Tarallini biscuits, comprising 1225 samples that are divided into four classes:
1. No defect (474 images)
2. Defect: not complete (465 images)
3. Defect: strange object (158 images)
4. Defect: color defect (128 images)
To augment the dataset, each sample was rotated three times by 90°, resulting in a total of 4900 images. A Python script was provided to prepare the Industrial Biscuits dataset and divide it into the required directory structure for training, validation, and testing. This script processes the original images and annotations, splitting them into separate folders based on the specified parameters.
The script performs the following steps:
1. Import necessary libraries (PIL for image processing, pandas for data manipulation, os for file handling, and math for floor function).
2. Define the paths for image, annotations, and destination folders.
3. Set the desired parameters for the dataset split: number of training OK samples, number
of training NOK samples, number of validation OK samples, number of validation NOK
samples, number of test OK samples, and number of test NOK samples.
4. Determine the defect ratios for the three types of defects: not complete, strange object,
and color defect.
5. Initialize counters for each category.
6. Calculate the number of samples for each defect category based on the defect ratios.
7. Create the dataset's directory structure if it does not already exist, including separate
folders for training, validation, and testing samples, with 'ok' and 'nok' subfolders in each.
8. Load the filenames and annotations from the CSV file.
9. Iterate over the dataset, accounting for the augmentation.
10. Open each image and check its annotation to determine its category (OK or one of the
three types of defects).
11. Save the image to the appropriate folder (train, valid, or test) and corresponding subfolder
(ok or nok) based on the category and counters.
12. Display a message indicating the successful creation of the dataset or the existence of the
folder structure.
This script effectively preprocesses the dataset and organizes it into a structure suitable for training a machine learning model using the Keras ImageDataGenerator or similar data-loading techniques.
Initially, the script processed only a subset of the entire dataset, which consists of 4900 images. However, the script has been modified to handle the entire dataset and split it based on the given
 
train (70%), validation (15%), and test (15%) ratios. It also calculates the total number of OK and NOK images in the dataset and the number of images in each category for train, validation, and test sets. Lastly, it prints the dataset statistics, including total images, the number of images in each set, and defect ratios. Table 1 and Figure 1 in the appendix illustrate dataset statistics organized for a comprehensive understanding of the dataset.

## Machine Learning Task

The primary machine learning task performed on the Industrial Biscuits dataset is multi-class classification. Utilizing the image data, the objective is to accurately categorize biscuits into one of the four predefined classes, which represent different defect types and the absence of defects. These classes are as follows:
1. No defect: Biscuits that meet the quality standards and exhibit no anomalies.
2. Defect: not complete: Biscuits with missing or incomplete parts, indicating potential
issues during the production process.
3. Defect: strange object: Biscuits contaminated with foreign objects or materials, raising
concerns about product safety and quality.
4. Defect: color defect: Biscuits exhibiting discoloration or inconsistent coloration, which
may indicate baking irregularities or ingredient inconsistencies.
By leveraging advanced machine learning techniques, such as deep learning and convolutional neural networks, the model aims to accurately classify biscuits into these categories, enabling efficient and reliable identification of potential quality control issues in the biscuit production process.

## Machine Learning Methods

The machine learning task at hand required a robust model capable of identifying and classifying anomalies in biscuit images. To achieve this, we leveraged the power of the Xception architecture and the EfficientNet family of models, both advanced deep convolutional neural network (CNN) architectures designed for high performance in tasks such as ours.
The Xception architecture, standing for "Extreme Inception," was proposed by François Chollet, the creator of the Keras library, in 2017. This architecture builds upon the Inception architecture with a key innovation: the use of depthwise separable convolutions in place of the standard convolutions used in most CNN architectures. Depthwise separable convolutions factorize the standard convolution operation into two separate operations: depthwise convolution and pointwise convolution. This factorization results in fewer parameters and computations, leading to more efficient training and inference, which was crucial given the complexity of our task. (Refer to figure 2 and 3 in the appendix)
  
The EfficientNet family of models, introduced by Mingxing Tan and Quoc V. Le in 2019, is another set of CNN architectures that have gained widespread popularity due to their efficiency and performance. These models use a compound scaling method that jointly scales the width, depth, and resolution of the networks. EfficientNets also employ mobile inverted bottleneck convolution (MBConv) and squeeze-and-excitation (SE) modules to optimize the network's efficiency. Our experiments utilized the EfficientNetB0 and EfficientNetB4 architectures to benchmark their performance against the Xception model. (Refer to figure 4 in the appendix)
Both the Xception and EfficientNet models were compiled with the Adam optimizer, using sparse categorical crossentropy as the loss function and accuracy as the evaluation metric. The models were trained on an augmented version of the dataset, which was split into training, validation, and test sets. Data augmentation techniques, such as rotation, width shift, height shift, zoom, and horizontal flip, were used to increase the robustness of the models to unseen data.
The hardware and software computing environment used for this task was as follows: Hardware:
- CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
- GPU: NVIDIA GeForce RTX 3090
Software:
- Platform: Linux-64
- Conda version: 4.12.0
- Python version: 3.9.12.final.0
- Keras: 2.3.1
- TensorFlow-GPU: 2.1.0
- Numpy: 1.19.5
- Pandas: 1.3.5
- Scikit-learn: 1.0.2
This configuration provided the necessary computational power and software tools to carry out our extensive experiments. The combination of powerful CNN architectures like Xception and EfficientNet, robust hardware, and reliable software libraries allowed for the development and training of effective models for the classification of industrial biscuits. The models' performance was evaluated using accuracy, precision, recall, and the F1-score, providing a comprehensive evaluation of their ability to correctly classify biscuits into the four categories.

## Machine Learning Pipeline

Our machine learning pipeline for this multi-class classification task was as follows:
1. Data Collection and Preprocessing: The dataset was collected and split into training, validation, and test sets. Images were preprocessed by rescaling pixel values from 0-255 to 0-1, a common practice in deep learning tasks involving images.
2. Data Augmentation: To improve the model's robustness and generalization capabilities, data augmentation techniques were applied to the training set. These techniques included rotation, width shift, height shift, zoom, and horizontal flipping of images. The augmented data was then fitted using the ImageDataGenerator function from Keras.
3. Model Creation: We used the Xception and EfficientNet architecture as the base for our model. The model starts with an Input layer that accepts images of shape (256, 256, 3). This is followed by the Xception base model, a GlobalAveragePooling2D layer, a Dense layer with 128 units and ReLU activation, a Dropout layer with a rate of 0.2 for regularization, and a final Dense layer with units equal to the number of classes (4 in our case) and softmax activation for multi-class classification.
4. Model Compilation: The model was compiled using the Adam optimizer with a learning rate of 0.0001. We used sparse categorical crossentropy as the loss function, and accuracy as the metric for performance evaluation.
5. Model Training: The model was trained for 30 epochs with a batch size of 32. Training was performed using the augmented data, and the validation data was used for evaluating the model after each epoch. The training process was monitored using the TqdmCallback to display a progress bar for each epoch.
6. Model Evaluation: The trained model was then evaluated on the test set. Predictions were obtained by taking the argmax of the predicted probabilities. The model's performance was then assessed using accuracy, precision, recall, and F1-score. These metrics provide a comprehensive evaluation of the model's performance across all classes.
7. Saving the Model and Training Progress: The model and its training progress across epochs were saved for future use and analysis. The model was saved as "my_model.h5" using the save method, and the training progress was saved as a CSV file named "training_progress.csv". This includes the loss and accuracy for both the training and validation sets for each epoch.
8. Replicating the Experiments: To replicate these experiments, the required software and hardware environment must be set up as specified. The same dataset and the provided source code should be used. Note that the results may slightly vary due to the inherent randomness in training deep learning models, even when the random seed is fixed.
This pipeline provides a comprehensive overview of the steps involved in our experiments.
 
## Experiments and Results

Various configurations were explored during the experimentation phase to optimize the performance of the model. These configurations included adjustments to the number of epochs, batch size, and learning rate. The model's performance was evaluated using accuracy, precision, recall, and F1-score on the test set.
Initially, the model was trained for 30 epochs with a batch size of 32 and a learning rate of 0.001. Subsequently, the learning rate was adjusted to 0.0001 while keeping the other parameters constant. The model achieved the following results:
• Learning rate = 0.001:
o Accuracy: 0.9957 o Precision: 0.9957
o Recall: 0.9957
o F1-score: 0.9957 • Learning rate = 0.0001:
o Accuracy: 0.9978
o Precision: 0.9979
o Recall: 0.9978
o F1-score: 0.9979 (Refer to figure 5 in the appendix)
Comparing the two learning rates, the model performed better with a learning rate of 0.0001 in terms of evaluation metrics. The training progress graph, which displayed loss and accuracy on the training and validation sets, also exhibited less fluctuation and faster convergence with the lower learning rate. (Refer to figure 6 in the appendix).
Further experimentation was conducted by increasing the batch size to 64 while maintaining a learning rate of 0.0001 and 30 epochs. This configuration yielded the following results:
• Batch size = 64:
o Accuracy: 0.9957
o Precision: 0.9957
o Recall: 0.9957
o F1-score: 0.9957 (Refer to figure 7 and 8 in the appendix)
A comparison of the different configurations reveals that the model achieved the best performance with 30 epochs, a learning rate of 0.0001, and a batch size of 32.
The performance of the fine-tuned Xception model was benchmarked against another popular deep learning architecture, EfficientNetB0, with equivalent hyperparameter configurations. The results on the testing set were as follows:
 
• EfficientNetB0:
o Accuracy: 0.9656
o Precision: 0.9684
o Recall: 0.9656
o F1-score: 0.9658 (Refer to figure 9 in the appendix)
In comparison, the fine-tuned Xception model outperformed the EfficientNetB0 architecture, indicating its effectiveness in this particular anomaly detection task.
Further experimentation was performed using a larger version of EfficientNet, EfficientNetB4. The results obtained were as follows:
• EfficientNetB4:
o Accuracy: 1.0
o Precision: 1.0
o Recall: 1.0
o F1-score: 1.0 (Refer to figure 9 and 10 in the appendix)
Thus, EfficientNetB4 shows the best result since it achieved an accuracy of 100%. Considering the superior performance of EfficientNetB4, there is no need to use a larger model or further hyperparameter tuning, data preprocessing, or data augmentation.
In conclusion, the experiments demonstrated the effectiveness of the fine-tuned Xception model in the anomaly detection task. However, the best performance was achieved using EfficientNetB4, which reached 100% accuracy, precision, recall, and F1-score on the test set.

## Discussion and Conclusion

Our experimentation with different machine learning models on the Industrial Biscuits dataset has led to noteworthy results. The fine-tuned Xception model performed admirably, demonstrating high metric scores and an impressive accuracy rate of over 99%. However, upon further experimentation with the EfficientNet family of models, the EfficientNetB4 model ultimately emerged as the most performant.
The EfficientNetB4 model achieved perfect scores across all metrics, including accuracy, precision, recall, and the F1-score, underscoring its effectiveness in this specific anomaly detection task. Moreover, the EfficientNetB4 model also exhibited the least fluctuation during the training process, indicating a stable and robust learning procedure.
While the EfficientNetB4 model has proven to be an excellent fit for our task, it is important to note that there are other more complex models available within the Keras library (Refer to figure 11 in the appendix). Some of these models, with larger architectures, have demonstrated superior performance on datasets like ImageNet. However, the efficiency of the EfficientNet models is what sets them apart. Achieving perfect evaluation metrics scores with the EfficientNetB4 model
 
indicates that more complex models may not necessarily yield better results for our task, and may require more computational resources and longer training times, which could be a significant consideration for real-world, industrial applications.
There are several potential areas for further exploration and improvement in future work:
1. Exploring other deep learning architectures or further refining the EfficientNetB4 model. While our results are promising, the landscape of deep learning is constantly evolving, and new models and techniques may offer additional improvements.
2. Experimenting with different data augmentation techniques or hyperparameter configurations. These techniques can potentially enhance the model's generalization capability and performance.
3. Investigating unsupervised or semi-supervised learning techniques for anomaly detection. These techniques can potentially improve the model's ability to identify novel anomalies that do not resemble any examples in the training set.
In conclusion, this project underscores the immense potential of deep learning techniques for anomaly detection tasks in industrial applications, specifically in biscuit production. The fine- tuned EfficientNetB4 model demonstrated superior performance and stability, suggesting that carefully customized deep learning models can effectively improve quality control processes in manufacturing settings by accurately detecting anomalies, while maintaining high efficiency.