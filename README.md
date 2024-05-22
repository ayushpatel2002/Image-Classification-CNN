# Road Traffic Sign Classification

<details>
<summary><h2>Introduction</h2></summary>
The aim of this project is to train at least two models to classify images of European road traffic signs based on their sign shape or sign type. To achieve this, I analyzed a specifically prepared dataset from the Belgium Traffic Sign Classification (TSC) Benchmark, consisting of 3699 grayscale images, using Jupyter Notebook. Exploratory Data Analysis (EDA) was performed to better understand the data composition. An ‘unseen’ independent evaluation dataset, compiled from a combination of images from the German TSC dataset, images from the internet, and digital photos taken personally from an iPhone, was used to evaluate each developed model's performance.

Two different supervised machine learning algorithms were investigated: Artificial Neural Network (ANN) and Convolutional Neural Network (CNN). Model optimization techniques such as hyperparameter tuning, regularization, dropout, data augmentation, and edge detection filtering were also explored. Based on the results obtained from each model's performance against the evaluation dataset, the best model for each task was identified.
</details>

<details>
<summary><h2>Training & Validation Dataset</h2></summary>
The dataset provided for this project is a modified version of the Belgium Traffic Sign Classification Benchmark which contains images of European road traffic signs taken from real-world vehicles. It consists of 3699 grayscale ‘.png’ images, each having a 28 (H) x 28 (W) pixel dimension.

Images have been placed in sub-directories in the format of `./trafficsigns_dataset/{sign-shape}/{sign-type}/`, corresponding to its correct classification. For example, an image in the `/diamond/rightofway/` directory has a sign shape of ‘diamond’, and a sign type of ‘rightofway’. Some sign types have different individual signs, such as the ‘speed’ sign type having signs from 10-70mph.
</details>

<details>
<summary><h2>Exploratory Data Analysis</h2></summary>

### Data Preparation
In preparing the dataset for analysis and model training, I utilized a script to systematically traverse through the directory structure of the traffic sign images. The script loops through each subdirectory, reading each image file (in PNG format), and extracting relevant metadata such as the image's file path, sign shape, sign type, dimensions, and an MD5 hash. The MD5 hash is used in a later step to ensure none of the images are duplicated in the evaluation set. These details are appended to a list, which is then converted into a pandas DataFrame. This DataFrame provides a structured view of the dataset, containing columns for the image's file path, image path, filename, sign shape, sign type, and dimensions. The final DataFrame consists of 3699 rows with 7 attributes each, enabling efficient EDA.

### Data Distribution
Checking the number of images under each shape revealed noticeable skewness in the distribution of images across different traffic sign shapes. The majority of the images are round signs, with 1760 images, comprising approximately 47.6% of the dataset. Triangle signs follow with 926 images (25%), and square signs with 688 images (18.6%). Diamond-shaped signs are less common, with 282 images (7.6%), and hexagonal signs are the least represented, with only 43 images (1.2%). This skewness indicates a higher concentration of certain sign shapes, particularly round and triangle, which could impact the model's performance and may necessitate strategies to handle the imbalance.

Checking the number of images under each sign type also reveals substantial skewness. The "warning" sign type is the most prevalent, with 695 images, making up approximately 18.8% of the dataset. The next most common types are "noentry" (375 images, 10.1%) and "speed" (316 images, 8.5%). In contrast, several sign types, such as "stop" (43 images, 1.2%), "crossing" (95 images, 2.6%), and "roundabout" (98 images, 2.7%), have relatively few images. This uneven distribution indicates that the dataset is heavily weighted towards certain sign types, and I will address class imbalance in the modeling process to ensure fair and effective training.

### Image Size Distribution
The dataset consists of images uniformly sized at 28x28 pixels. This uniformity ensures consistency in the input data for image processing and model training, simplifying preprocessing steps such as resizing and normalization. The fact that all 3699 images adhere to this dimension is critical for maintaining a standardized input format for the convolutional neural network (CNN) models used in the classification tasks. This consistency helps streamline the workflow and ensures that the models can efficiently process the data without additional adjustments for varying image sizes.

### Colour Analysis
The dataset consists of grayscale images, where each color channel (red, green, and blue) carries the same intensity values, resulting in identical distributions across all channels. The pixel intensities range from 0 to 255, with higher concentrations in the lower range (around 50-100) and near the maximum intensity (255). This indicates a variety of shades, with significant amounts of both darker and brighter pixels. This uniform grayscale representation ensures consistency in the dataset, simplifying preprocessing and feature extraction for model training.

### Data Splitting
Since I am testing my model with real-world data that would be completely unseen, I have only split it into train and validation sets in 80-20 percent respectively. After splitting, the training data consists of 2959 images and the validation data consists of 740 images.

### Data Leakage Check
The distribution of sign shapes and sign types in the training and validation sets demonstrates good overlap, showing that both sets contain similar proportions of each type. The consistent peaks for common shapes and types across both datasets confirm good representation and suggest that the validation set can provide a realistic indication of the model's performance on new, unseen data. The data split appears to be effective, suggesting that the validation set mirrors the training data well, supporting effective model training and validation.
</details>

<details>
<summary><h2>Evaluation Metrics</h2></summary>

### Chosen Metrics: Accuracy and Weighted F1-Score
Accuracy serves as a general performance indicator, measuring the overall correctness of the model. It is defined as the ratio of correctly predicted observations to the total observations, providing a quick and intuitive measure of the model’s general effectiveness.

The Weighted F1-Score addresses the class imbalance present in the dataset. It adjusts the F1-Score for each class by the number of true instances, giving more weight to classes with more samples. This metric provides a balance between precision and recall in a single number, weighted by the class distribution. This ensures that the model’s performance on less common classes significantly influences the overall score.
</details>

<details>
<summary><h2>Model Development</h2></summary>

### Artificial Neural Network (ANN) - Baseline
Starting with an ANN allows me to establish a straightforward performance metric to determine if the complexity added by Convolutional Neural Networks (CNNs) is justified. This approach is particularly useful because it allows me to quickly identify any major issues or bottlenecks in the initial model without the added complexity of a CNN. Additionally, ANNs are faster to train and easier to debug, making them ideal for initial experimentation and validation.

### ANN Architecture
1. **Flatten Layer**: Transforms the 2D 28x28 pixel grayscale images into a 1D vector of 784 elements, necessary for dense layers in a neural network that accept vectors as input.
2. **Dense Layer (Hidden Layer)**: The hidden layer with 256 neurons captures complex relationships and features from the flattened input vector using ReLU activation.
3. **Output Layer**: Has 5 or 16 neurons, corresponding to the five classes of sign shapes or sixteen classes for sign types. The layer’s output (logits) is directly utilized by the loss function due to `from_logits=True`.

### Compilation Settings
1. **Optimizer**: Adam, an adaptive learning rate optimization algorithm that can handle sparse gradients on noisy problems.
2. **Loss Function**: CategoricalCrossentropy with `from_logits=True` for multi-class classification tasks.
3. **Metrics**: Categorical accuracy to evaluate the performance of the model in a multi-class classification scenario.

### Convolutional Neural Network (CNN)
#### Sign Shape Architecture
- **Input Layer**: Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)) with 32 filters of size 3x3.
- **Pooling Layer**: MaxPooling2D((2, 2)).
- **Dropout Layer**: Dropout(0.25).
- **Second Convolutional Layer**: Conv2D(64, (3, 3), activation='relu') with 64 filters.
- **Second Pooling Layer**: MaxPooling2D((2, 2)).
- **Second Dropout Layer**: Dropout(0.25).
- **Flattening Layer**: Converts the 3D output into a 1D vector.
- **Dense Layer**: Dense(128, activation='relu').
- **Third Dropout Layer**: Dropout(0.5).
- **Output Layer**: Dense(5, activation='softmax') for multi-class classification.

#### Sign Type Architecture
- Similar to Sign Shape Architecture with the addition of a third convolutional layer (Conv2D(128, (3, 3), activation='relu')) and corresponding pooling and dropout layers.
- **Output Layer**: Dense(16, activation='softmax') for multi-class classification.
### Model Compilation
- **Optimizer**: Adam for efficient weight adjustment.
- **Loss Function**: Categorical crossentropy for multi-class classification.
- **Metrics**: Accuracy to monitor the percentage of correctly predicted instances.

### Detection Filter
Used Sobel edge detection filter in pre-processing to identify edges in each image before model training.

### Hyperparameter Tuning
Conducted using Bayesian search on multiple models across both ANN and CNN. For ANN models, the number of neurons in the first hidden layer and the learning rate were tuned. For CNN models, filters, kernel size, kernel regularizer, dropout rate, and dense layer units were tuned.

### Early Stopping
Incorporated to terminate training when the selected metric no longer improves within a chosen ‘patience’ period, reducing overfitting.
</details>

<details>
<summary><h2>Evaluation Dataset</h2></summary>
An ‘unseen’ independent evaluation dataset, consisting of 493 images, was used to evaluate the performance of each developed model. This dataset was compiled from a combination of the following:
- Images from the German TSC dataset
- Images from the test set of the Belgium TSC dataset
- Other images taken from the internet
- Digital photos taken personally from an iPhone

A significant portion of the evaluation dataset was sourced from the German TSC dataset due to the similarity in street sign design between Germany and Belgium. The Belgium TSC dataset was also used due to the absence of a few sign types in the German TSC dataset. Additionally, images sourced from the internet were used to increase the sample size for certain sign types. Personally taken digital photos were included to ensure the models are viable in real-world scenarios. Prior to evaluation, the dataset was preprocessed by converting images to grayscale, resizing to 28x28, and applying an edge detection filter (if applicable).
</details>

<details>
<summary><h2>Final Evaluation Results</h2></summary>
The results (based on the evaluation metrics chosen) of each model architecture against variations of the training dataset are summarized below. Note:
- ‘_opt’ refers to an optimized model with hyperparameter tuning conducted.
- Data Augmentation applied only on the training set.
- Filter applied on training, validation, and evaluation sets.

### Sign Shape
| Model              | Original Data  | Data Augmentation | Data Aug + Filter |
|--------------------|----------------|-------------------|-------------------|
| ANN shape_ann_1    | Acc: 0.73      | Acc: 0.62         | Acc: 0.87         |
|                    | WF1: 0.73      | WF1: 0.62         | WF1: 0.87         |
| ANN shape_ann_opt* | Acc: 0.70      | Acc: 0.81         | Acc: 0.86         |
|                    | WF1: 0.69      | WF1: 0.78         | WF1: 0.85         |
| CNN shape_cnn_1    | Acc: 0.96      | Acc: 0.97         | Acc: 0.97         |
|                    | WF1: 0.95      | WF1: 0.97         | WF1: 0.97         |
| CNN shape_cnn_opt* | Acc: 0.97      | Acc: 0.98         | Acc: 0.98         |
|                    | WF1: 0.95      | WF1: 0.98         | WF1: 0.98         |
| CNN shape_cnn_google| Acc: 0.90     | Acc: 0.95         | Acc: 0.97         |
| net                | WF1: 0.90      | WF1: 0.95         | WF1: 0.96         |

Based on the results, the best model for sign shape is the ‘shape_cnn_opt’ model with data augmentation, achieving an accuracy and weighted F1-score of 0.98.

### Sign Type
| Model               | Original Data  | Data Augmentation | Data Aug + Filter |
|---------------------|----------------|-------------------|-------------------|
| ANN type_ann_1      | Acc: 0.51      | Acc: 0.62         | Acc: 0.74         |
|                     | WF1: 0.51      | WF1: 0.62         | WF1: 0.73         |
| ANN type_ann_opt*   | Acc: 0.61      | Acc: 0.68         | Acc: 0.78         |
|                     | WF1: 0.61      | WF1: 0.68         | WF1: 0.77         |
| CNN type_cnn_tl     | Acc: 0.78      | Acc: 0.80         | Acc: 0.78         |
|                     | WF1: 0.77      | WF1: 0.79         | WF1: 0.76         |
| CNN type_cnn_1      | Acc: 0.87      | Acc: 0.90         | Acc: 0.90         |
|                     | WF1: 0.86      | WF1: 0.90         | WF1: 0.89         |
| CNN type_cnn_opt*   | Acc: 0.82      | Acc: 0.88         | Acc: 0.84         |
|                     | WF1: 0.81      | WF1: 0.88         | WF1: 0.83         |
| CNN type_cnn_googlenet| Acc: 0.74    | Acc: 0.88         | Acc: 0.83         |
|                     | WF1: 0.74      | WF1: 0.88         | WF1: 0.83         |

The best model for sign type is the ‘type_cnn_1’ model with data augmentation, achieving an accuracy and weighted F1-score of 0.90.
</details>

<details>
<summary><h2>Discussion</h2></summary>
Due to the increase in hyperparameters requiring tuning in CNN models, a larger number of trials were required to arrive at the optimal values compared to ANN where only two hyperparameters were tuned. As a result, the performance of ‘optimized’ models may be lower due to suboptimal hyperparameter values, as seen in the case of ‘type_cnn_opt’ compared to ‘type_cnn_1’.

### Edge Detection Filters
- **ANN Performance Improvement**: Edge detection filters simplify input data by highlighting edges and removing extraneous details, benefiting ANNs by making it easier to learn important features.
- **CNN Performance Decrease**: Edge detection filters can remove important contextual and textural information, leading to decreased performance as CNNs rely on rich, unaltered input data to learn and apply their own filters.
</details>

<details>
<summary><h2>Conclusion</h2></summary>
In conclusion, appropriate machine learning techniques were selected and applied to create two models used to predict the sign shape or sign type of European Traffic Street signs. EDA was performed on the datasets to better understand the data in depth, such as image dimension, class imbalance, and color distribution. An independent evaluation set was created from various sources, including real-world photos taken personally to mimic the real-world feasibility of the models created, ensuring no overlap between the evaluation set and the training or validation dataset.

ANN and CNN architectures were chosen for model development, with further optimization conducted such as hyperparameter tuning, data augmentation, edge detection filtering, early stopping, and other regularization techniques to reduce overfitting and improve model performance. After analyzing the result of each model variation, the final model chosen for sign shape prediction was ‘shape_cnn_opt’ trained with data augmentation, achieving an accuracy and weighted F1-score of 0.98. The chosen model for sign type was ‘type_cnn_1’ trained with data augmentation, achieving an accuracy and weighted F1-score of 0.90.
</details>

<details>
<summary><h2>References</h2></summary>
1. Ahmad, I. (2024). ‘Sigmoid vs ReLU’, Educative. Retrieved May 5, 2024, from https://www.educative.io/answers/sigmoid-vs-relu.
2. Vishwakarma, N. (2023). ‘What is Adam Optimizer?’, Analytics Vidhya. Retrieved May 5, 2024, from https://www.analyticsvidhya.com/blog/2023/09/what-is-adam-optimizer/.
3. Sharma, N., et al. (2018). “An Analysis Of Convolutional Neural Networks For Image Classification.” Procedia Computer Science, vol. 132, pp. 377–84, https://doi.org/10.1016/j.procs.2018.05.198.
4. Yamashita, R., Nishio, M., Do, R.K., & Togashi, K. (2018). ‘Convolutional neural networks: an overview and application in radiology’, Insights into Imaging, vol. 9, no. 4, pp. 611-629. Retrieved May 18, 2024, from https://doi.org/10.1007/s13244-018-0639-9.
5. Heaton, J. (2018). “Ian Goodfellow, Yoshua Bengio, and Aaron Courville: Deep Learning: The MIT Press, 2016, 800 Pp, ISBN: 0262035618.” Genetic Programming and
