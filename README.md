# Practical-Assignment-17

## Project Name - "Comparing Classifiers"

**Business Understanding of the Problem:**

This is the third practical application assignment, and its primary objective is to compare the performance of various classification algorithms — including k-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines — introduced throughout the program. For this purpose, we will utilize a dataset related to the telephone-based marketing of bank products. To gain deeper insights into the problem we aim to address, it is essential to first understand the origin of the data and its relevance in real-world scenarios. 

**Source Dataset and Insights on dataset:**

The dataset used in this project is sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) . It originates from a Portuguese banking institution and comprises data collected from multiple marketing campaigns aimed at promoting term deposit subscriptions. The dataset includes various client and campaign-related attributes, making it well-suited for classification and predictive modeling tasks. For more information, we have used the article accompaying the dataset [here](https://github.com/PoojaSinha8809/Practical-Assignment-17/blob/main/CRISP-DM-BANK.pdf)

In the project, we have used the dataset located here - "data/bank-additional/bank-additional-full.csv". This bank-additional-full.csv contains all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed. 

With reference to paper study, we are going to use CRISP-DM methodology. This dataset contains real-world data collected from a series of marketing campaigns conducted by a Portuguese banking institution, specifically aimed at promoting term deposit subscriptions. The primary business objective is to develop a model that can predict the success of a contact — that is, whether a client will subscribe to a deposit. Such a model can significantly enhance campaign efficiency by identifying the key factors influencing customer decisions. This, in turn, supports better allocation of resources, such as time, personnel, and communication efforts, and enables the selection of a more targeted and cost-effective group of potential customers.

The techiqiue we used in reference to the paper follows CRISP DM framework, which can be find below: 

![CRISP DM Framework](images/screenshots/img1.jpg)

## Key Steps in the Project:
    
### Data Understanding and Preprocessing: 
In this phase, we begin by loading the CSV file into a Pandas DataFrame to facilitate data analysis and exploration. This allows us to examine the relationships between various features in the dataset and understand their potential impact on client decisions. As part of the data preprocessing, we check for and address duplicate entries, missing values, and any inappropriate or inconsistent data. Ensuring data quality is essential for building robust and accurate models. Additionally, we employ various data visualization techniques to explore feature distributions and identify patterns or trends that may influence a client's decision to subscribe to a term deposit. These visual insights help guide the selection of relevant features for modeling.

**Understanding the features:**

In the data exploration, we find that there is no missing value in the dataset. So based on features data we have categorized the data further. 
Data Categorization

Here’s how the data can be categorized into Client Profile, Marketing Data, and Output Variable:

**Client Profile (Personal and Socioeconomic Data): This category includes data that describes the client’s personal attributes, demographic information, and their socioeconomic context.**
- Age (numeric) – Age of the client.
- Job (categorical) – Type of job (e.g., 'admin.', 'blue-collar', 'entrepreneur', etc.).
- Marital (categorical) – Marital status (e.g., 'divorced', 'married', 'single', etc.).
- Education (categorical) – Level of education (e.g., 'basic.4y', 'university.degree', etc.).
- Default (categorical) – Whether the client has credit in default ('no', 'yes', 'unknown').
- Housing (categorical) – Whether the client has a housing loan ('no', 'yes', 'unknown').
- Loan (categorical) – Whether the client has a personal loan ('no', 'yes', 'unknown').


**Marketing Data (Campaign and Contact Information): This category includes data related to the client’s interactions during the current and previous marketing campaigns.**

- Contact (categorical) – Communication type used for contact ('cellular', 'telephone').
- Month (categorical) – Month of the last contact ('jan', 'feb', ..., 'dec').
- Day of Week (categorical) – Day of the week of the last contact ('mon', 'tue', ..., 'fri').
- Duration (numeric) – Duration of the last contact in seconds (Important note: should only be included for benchmarking purposes as it is not  available before the call).
- Campaign (numeric) – Number of contacts made during the current campaign for this client (includes the last contact).
- Pdays (numeric) – Number of days since the client was last contacted in a previous campaign (999 means no previous contact).
- Previous (numeric) – Number of contacts made before this campaign for this client.
- Poutcome (categorical) – Outcome of the previous campaign ('failure', 'nonexistent', 'success').
- Output Variable (Target Variable): This category includes the desired outcome of the campaign that we aim to predict.
- Y (binary) – Has the client subscribed to a term deposit? ('yes', 'no').


**Social and Economic Context Data: This category includes broader economic and social context indicators that might influence campaign success but are not specific to the client.**

- Emp.var.rate (numeric) – Employment variation rate (quarterly indicator).
- Cons.price.idx (numeric) – Consumer price index (monthly indicator).
- Cons.conf.idx (numeric) – Consumer confidence index (monthly indicator).
- Euribor3m (numeric) – Euribor 3-month rate (daily indicator).
- Nr.employed (numeric) – Number of employees (quarterly indicator).

**Summary of Categories:**

- Client Profile: Attributes related to the individual’s personal and socioeconomic information.
- Marketing Data: Attributes related to the client’s interaction in the campaign and previous campaign outcomes.
- Output Variable: The target we aim to predict, i.e., whether the client subscribed to the term deposit.
- Social and Economic Context Data: Broader contextual data that may influence campaign outcomes.

This categorization helps organize the data based on its role in the predictive modeling process and ensures a clearer structure for analyzing the factors contributing to the success of the marketing campaign.


### Splitting the Data into Training and Testing: 

We will begin by splitting the dataset into training and testing subsets. This is a crucial step in building a reliable machine learning model, as it allows us to train the model on one portion of the data and evaluate its performance on unseen data. The split ratio between training and testing sets can be adjusted based on the specific needs of the analysis, though a common practice is to use a standard split such as 70/30 or 80/20. Choosing an appropriate split ensures that the model has enough data to learn meaningful patterns while still being evaluated fairly on data it has not encountered during training.


### Basic Model Development: 

Once the dataset is split into training and testing sets, the next critical step is to build baseline classification models using a variety of algorithms. In this assignment, we focus on four widely used machine learning classifiers: Decision Tree, k-Nearest Neighbors (k-NN), Support Vector Machine (SVM), and Logistic Regression. Each of these models offers unique strengths and approaches to classification tasks. The goal of building these baseline models is to evaluate their initial performance on the dataset, using consistent metrics such as accuracy, precision, recall, and F1-score. This comparison helps identify which algorithm is most suitable for the given problem. Moreover, establishing baseline results sets the foundation for further model tuning and optimization in later stages, such as hyperparameter tuning, feature selection, or ensemble techniques. By systematically evaluating and comparing these models, we can make informed decisions about which classifier best balances performance, complexity, and interpretability for the task at hand. 

### Model Evaluation and Comparision 
Model evaluation in classification tasks is a crucial step to assess how well a model performs in predicting categorical outcomes. It involves using a variety of metrics that go beyond simple accuracy, providing a more comprehensive view of model performance. Common evaluation metrics include the confusion matrix, precision, recall, F1-score, and accuracy. Accuracy measures the overall correctness of the model, but it can be misleading when dealing with imbalanced datasets. Precision focuses on the proportion of correct positive predictions, while recall measures the model's ability to identify all relevant positive cases. The F1-score, as the harmonic mean of precision and recall, offers a balanced evaluation metric, especially useful when false positives and false negatives carry different consequences. Additionally, techniques such as cross-validation help ensure the model's performance is consistent and not just a result of overfitting to a particular dataset. Proper evaluation allows for informed decision-making when selecting and fine-tuning classification models for real-world applications.


### Findings on Data Analysis:

