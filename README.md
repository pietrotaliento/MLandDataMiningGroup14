#Coursework 1 for Machine Learning and Data Mining F21DL by group 14 Postgraduates Dubai
Nakhul Kalaivanan 
Goktug Peker
Pietro Taliento
Tabular Dataset:
Give Me Some Credit 
What is the dataset about?
This dataset, available on Kaggle, is centered around predicting financial distress of individuals based on various financial and demographic attributes. This dataset is highly relevant in financial analysis and modeling, as it helps in evaluating the creditworthiness of customer
https://www.kaggle.com/competitions/GiveMeSomeCre dit/data 
What is the aim of the dataset
The aim of this dataset is to predict whether a person is likely to experience a serious financial delinquency in the range of two years. This enables financial institutions to evaluate the creditworthiness of each customer and make informed loaning decisions.

Why did we choose this dataset?
● Complementary Nature: The "Give Me Some Credit" dataset provides a rich set of features that can complement the "Sign Language MNIST" dataset's image data, which we’ll see later. By combining an image dataset and a tabular dataset, we can present a diverse range of data analysis and machine learning models. Also, unlike the next dataset, which is perfectly labeled, this dataset presents some inherent challenges, such as missing values. These challenges are ideal for demonstrating different data preprocessing techniques and machine learning strategies
● Financial Relevance/Application to real world scenarios: Financial data analysis is a crucial application of machine learning and has a high impact on real-world business decisions. Therefore, this dataset covers a real-world scenario where banks or other financial institutions need to assess the creditworthiness of applicants, making it an excellent choice for demonstrating the practical applications of the models we implement.

 Features(1/2)
  ● Classes:
○ Class 0: Serious Delinquency
○ Class 1: Non-Serious Delinquency
● Instances: 101504 across all classes, each representing a customer ● Features:
○ RevolvingUtilizationOfUnsecuredLines: Total balance on credit cards and personal lines of credit relative to the credit limit.
○ age: Age of the borrower
○ NumberOfTime30-59DaysPastDueNotWorse: Number of times a borrower was overdue by 30-59
days but did not go into further default.
○ DebtRatio: Ratio of monthly debt payments, alimony.
○ MonthlyIncome: Monthly income of the borrower.
 Features (2/2)
  ○ NumberOfOpenCreditLinesAndLoans: Number of open loans and credit lines.
○ NumberOfTimes90DaysLate: Number of times a borrower has been overdue by 90 days or more.
○ NumberRealEstateLoansOrLines: Number of real estate loans or lines of credit (e.g., mortgage).
○ NumberOfTime60-89DaysPastDueNotWorse: Number of times a borrower was overdue by 60-89
days but did not go into further default.
○ NumberOfDependents: Number of dependents of the borrower. Data Visualization
● We started with a basic exploration of the dataset by extracting essential information such as data types, measurement levels, and the first five rows to confirm the integrity of the data
● Next, we conducted preliminary visualizations using histograms and scatter plots to gain a deeper understanding of the feature distributions and relationships
● To further refine our understanding, we performed feature engineering by creating new attributes and evaluating their importance 
Research questions
“How does ‘monthly income’ affect financial distress ?”
"Can clustering techniques such as K-Means be applicable for each person?”
“How does ‘monthly income’ affect financial distress ?”
Hypothesis: Amongst all the feature, ‘monthly income’ might be the most relevant predictors of the individual’s financial distress
Can clustering techniques such as K-Means be applicable for each person?”
Hypothesis: Clustering will successfully segment customers into low, medium, and high-risk categories, with distinct patterns in credit utilization and payment history. This segmentation can help the management of customers in a multiple number of ways.
Objectives
● Perform a complete analysis of the dataset to understand the distribution and structure of key financial variables
● Identify patterns and relationships among variables like income and debt levels to gain insights into the factors affecting creditworthiness.
● Explore the data using clustering algorithms to group customers based on similar situations 
In order to practice all the techniques of our coursework, we are also going to include a Dataset with Images
Image Dataset:
Sign Language MNIST
What is the dataset about?
This dataset is one of the latest adaptations of the MNIST (Modified National Institute of Standards and Technology) image dataset of handwritten digits, a global point of reference for evaluating computer vision models, especially those focused on image classification tasks. This dataset focuses on static hand gesture recognition of the American Sign Language alphabet (ASL).
Link:https://www.kaggle.com/datasets/datamunge/sign-lan guage-mnist
What is the aim of the dataset?
This dataset aims to cluster similar hand gestures and utilize Convolutional Neural Networks (CNNs) to classify different signs, and the final goal is the real-time recognition of American Sign Language letters. Why did we choose this dataset?
● Large volume of data: The dataset includes 27,455 labeled images for training and 7,172 images for testing, each with a resolution of 125 pixels as documented in the accompanying Excel file . This large volume of well-structured data allows for a specific analysis of sign language recognition.
● Relevance to Machine Learning Applications: Its structured nature and visual format provide an ideal foundation for exploring and understanding image classification techniques, making it a great educational resource for learning advanced ML concepts.
● Clear Social Impact: There is no need to explain how an effective sign language recognition system can contribute to accessibility and inclusion for the Deaf and Hard of Hearing community, making it a socially relevant project
● Stimulating challenge: The dataset contains images with narrow differences between similar signs, making it a good test for a model's precision  Detailed overview
   ● Classes: 24, representing the American Sign Language alphabet (J and Z are excluded because they involve movements)
● Instances: 27,455 images for training, 7,172 images for testing
● Features:
○ Pixels: an integer value between 0 and 255, indicating the intensity of the grayscale
○ Image dimensions: 28x28 pixels, however only 125 pixels per image contain significant
informations and compose the class
● Data Representation: Two CSV files (one for training, one for testing), where each row
corresponds to an instance and labels the 125 key pixel values for each image
● Use case example: building machine learning models that can recognize static ASL
gestures
           
 Data Visualization
   ● We have processed the data and done the visualization of the dataset.
● we found the info of the dataset having 7172 for test and train is 27454 
Research questions
“How can sign language recognition models generalize across different individuals with varying hand shapes, speeds, and styles?”
"Can a unified machine learning model be trained to recognize signs from multiple sign languages and differentiate between them?"
" How can machine learning algorithms be developed to automatically recognize and translate sign language gestures into text ?" 
“How can sign language recognition models generalize across different individuals with varying hand shapes, speeds, and styles?”
Hypothesis: Incorporating diverse training data with variations in hand shapes, signing speeds, and individual styles, combined with data augmenta<img width="739" alt="Screenshot 2024-11-23 at 2 42 30 PM" src="https://github.com/user-attachments/assets/a54d67a8-b9c6-4a11-a36a-1c2cd5fbbec2">
tion techniques and advanced machine learning models such as Convolutional Neural Networks (CNNs) with attention mechanisms or transformers, will result in a sign language recognition model that generalizes well across different individuals, maintaining high accuracy and robustness in diverse real-world settings.
"Can a unified machine learning model be trained to recognize signs from multiple sign languages and differentiate between them?"
Hypothesis: A unified machine learning model, incorporating multi-task learning and language-specific feature extraction, can be trained to recognize signs from multiple sign languages. By leveraging shared visual patterns across languages while distinguishing language-specific nuances, the model will achieve high accuracy in both recognizing signs and correctly identifying the corresponding sign language.
" How can machine learning algorithms be developed to automatically recognize and translate sign language gestures into text?"
Hypothesis: A machine learning system that combines Convolutional Neural Networks (CNNs) for visual feature extraction, Recurrent Neural Networks (RNNs) or Transformers for sequential modeling, and real-time optimization techniques can effectively recognize and translate sign language gestures into text or speech with high accuracy and minimal latency in real-time applications.
Objectives
● Learn the basics of how machine learning models classify images by recognizing patterns in hand gestures representing sign language letters.
● Apply basic machine learning algorithms like K-Nearest Neighbors (KNN) or Logistic Regression to classify the sign language gestures
● Implement CNNs, a more advanced model suited for image data, to recognize hand gestures more accurately by extracting features like edges and shapes

 R2 and R3
  ● EDA and feature selection(Tabular Dataset):
○ Handle missing values and data type conversion to prepare data
○ Perform summary statistics mean, median, standard deviation
○ histograms and scatter plots to visualize
○ Correlation Analysis
● Clustering(Tabular and Image Dataset): ○ Algorithms:
■ K-Means Clustering
■ Hierarchical Clustering ○ Evaluation:
■ Silhouette Score ■ Elbow Testing R4
  ● Baseline Training and Evaluation Experiments(Tabular and Image Dataset):
○ Machine Learning algorithms:
■ K-NN
■ Naive Bayes
■ Logistic Regression
■ Linear Regression
○ Model Evaluation metrics:
■ Accuracy
■ Precision and Recall
■ Confusion Matrix R5
● Neural Networks (Image Dataset)
○
○
○
Neural Network Models:
■ Multi-Layer Perceptron (MLP)
■ Convolutional Neural Networks (CNNs) with late activation Training and Fine-Tuning:
■ Early Stopping
■ Batch Normalization and Dropout Performance Evaluation:
■ Compare neural network performance to other ML models
■ Loss Curves
■ Confusion Matrix
