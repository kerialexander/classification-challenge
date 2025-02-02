# Email Spam Detection Classification Challenge
Module 13 Challenge

## Project Overview
This project focuses on enhancing the email filtering system for an Internet Service Provider (ISP) using machine learning techniques. The objective is to develop and compare two classification models—Logistic Regression and Random Forest—to accurately detect spam emails and prevent them from reaching customers' inboxes.

The project demonstrates the complete workflow for supervised machine learning, including data preparation, feature scaling, model training, and evaluation. By comparing the models, insights are gained into their performance and suitability for email spam detection.

This project evaluates the performance of different machine learning models in handling classification data. Specifically, it compares the **Random Forest Classifier Model** with the **Logistic Regression Model**, highlighting their respective capabilities in handling different types of relationships within the data.

### Random Forest vs. Logistic Regression
The **Random Forest Classifier Model** is advantageous in classification tasks because it can **capture non-linear relationships** in the data. In contrast, **Logistic Regression** assumes a **linear decision boundary**, making it less effective for complex patterns. This distinction explains why the Random Forest model tends to perform better in scenarios where relationships between features and labels are non-linear. For this reason, I predicted that the Random Forest Model would preform better for this classification challenge.

## What the Code Does
The Jupyter Notebook (`email_spam_detection_project.ipynb`) implements the following steps:

1. **Data Loading and Preparation**
   - Loads the dataset containing email attributes and spam classifications.
   - Extracts the labels (`y`) from the `spam` column (`0` for legitimate emails, `1` for spam).
   - Extracts the features (`X`) from the other columns representing email characteristics.

2. **Data Splitting**
   - Splits the dataset into training and testing sets to evaluate the model's performance on unseen data.

3. **Feature Scaling**
   - Uses `StandardScaler` to standardize features, ensuring uniform scaling, which improves model performance.

4. **Logistic Regression Model**
   - Trains a logistic regression model on the scaled training data.
   - Makes predictions on the testing data.
   - Evaluates the model's accuracy.

5. **Random Forest Model**
   - Trains a random forest classifier on the scaled training data.
   - Makes predictions on the testing data.
   - Evaluates the model's accuracy.

6. **Model Evaluation**
   - Compares the accuracy scores of the logistic regression and random forest models.
   - Determines which model performs better for spam detection.

## Dataset
The dataset includes:
- **`spam`**: The target label (0 = legitimate email, 1 = spam email).
- **Features**: Various characteristics of emails used for classification.

The dataset can be downloaded from the following link:  
[Spam Data CSV](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv)

## File References
The project includes the following files:

1. **Main Notebook**:  
   - [`email_spam_detection_project.ipynb`](./email_spam_detection_project.ipynb): Contains the complete implementation of data preparation, model training, and evaluation.

2. **Starter Code**:  
   - [`email_spam_detection_starter_code.ipynb`](./email_spam_detection_starter_code.ipynb): A template notebook to help set up the project.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/email-spam-detection.git
   cd email-spam-detection
   ```

2. Install necessary dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook email_spam_detection_project.ipynb

   ```
   
## Model Evaluation
Based on the evaluation of the models, the following insights were gained:

1. **Which model performed better?**
   - The Random Forest model outperformed the Logistic Regression model in terms of accuracy. This is due to the Random Forest's ability to handle complex patterns in the data by combining multiple decision trees.

2. **How does this compare to the initial prediction?**
   - The results matched the initial prediction, as Random Forest is generally expected to perform better in scenarios with complex datasets, owing to its ensemble nature and ability to reduce overfitting.
  
3. **Additional Observations:**
   - Logistic Regression showed reasonable performance but was less effective in capturing intricate patterns in the data.
   - Feature scaling improved the performance of both models, highlighting the importance of preprocessing.

### Why the Random Forest Model Performed Better

The **Random Forest Classifier Model** outperformed **Logistic Regression** in the spam detection task due to the following reasons:

- **Ability to Capture Non-Linear Relationships:** Random Forest can learn complex decision boundaries, whereas Logistic Regression is limited to linear separations.
- **Higher Accuracy:** The accuracy score was higher for Random Forest, indicating better overall performance.
- **Improved Precision, Recall, and F1-Score:** The classification report showed that Random Forest had better precision and recall, meaning it was more effective in minimizing false positives and false negatives.
- **Fewer Misclassifications:** The confusion matrix revealed that Random Forest had a lower misclassification rate compared to Logistic Regression.

### Final Analysis
These results confirm that **ensemble methods like Random Forest are superior when handling complex classification tasks**, making them a better choice for spam detection compared to traditional linear models.

## Project Insights
This project highlights:
- Practical data preprocessing techniques for machine learning tasks.
- The implementation and evaluation of logistic regression and random forest models.
- The role of feature scaling in improving model performance.
- Insights into the strengths and limitations of each model for spam detection.

This project serves as a comprehensive introduction to applying machine learning for real-world classification problems.

## Resources
I attended a tutoring session to obtain assistance with a few issues with the initial code. 
