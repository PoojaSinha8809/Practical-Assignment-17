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

### Findings on Data Analysis:

n our analysis, we utilized a variety of data visualization techniques to gain deeper insights into the dataset and support informed decision-making during model development. One of the key areas we explored was data balance, particularly in terms of class distribution, to determine whether the dataset was imbalanced — a common issue that can significantly affect the performance of classification models. Visualization tools such as bar plots or pie charts were used to clearly represent the proportion of each class (e.g., clients who subscribed vs. those who did not).

We also analyzed feature correlation using heatmaps and correlation matrices to identify relationships between numerical variables. This helped in detecting multicollinearity, which can affect certain models like Logistic Regression, and also in understanding which features might be redundant or highly related.

Furthermore, we explored the impact of individual features on the target variable using box plots, count plots, and scatter plots. These visualizations provided valuable insight into which attributes (e.g., age, job type, previous contact outcome) were more likely to influence a client’s decision to subscribe to a term deposit. Understanding these relationships was essential for feature selection and for improving the overall performance and interpretability of the classification models.

We conducted Exploratory Data Analysis (EDA) on both numerical and categorical features to better understand the structure and distribution of the data. For numerical variables, we examined summary statistics, distributions, and outliers using tools such as histograms, box plots, and correlation matrices. For categorical variables, we analyzed the frequency of each category, visualized relationships with the target variable using count plots and bar charts, and assessed their potential impact on model performance. This comprehensive EDA helped uncover hidden patterns, detect anomalies, and guide feature selection for building effective classification models.

**In EDA of Numerical Data, we got following result :**
![EDA Numerical Data](images/screenshots/img26.jpg)

If the values for emp.var.rate (employment variation rate) and cons.price.idx (consumer price index) have negative values, it's important to interpret them correctly, as these are indicators of economic trends.

Here’s what the negative values represent:

1. Emp.var.rate (Employment Variation Rate)

- Interpretation of Negative Values:

>> This variable measures the quarterly change in employment. A negative value indicates a decrease in employment or a decline in the number of employed individuals in that quarter.

For example, if emp.var.rate = -3.4, it means that employment has decreased by 3.4% in that quarter.

2. Cons.price.idx (Consumer Price Index)

- Interpretation of Negative Values:

>> The consumer price index (CPI) measures the average change in prices paid by consumers for goods and services. A negative value suggests a deflationary trend, meaning that overall prices are decreasing.

For example, if cons.price.idx = -50.8, this suggests that the consumer price index dropped by 50.8 points compared to the previous period.This could indicate a drastic decline in prices (deflation), often caused by a significant reduction in demand, a financial crisis, or other extreme economic conditions.

Why Negative Values Occur?

- Economic Downturn: Both of these indicators can be negative during periods of economic recession or downturn, reflecting issues like rising unemployment and declining prices.

- Deflation: A negative CPI (consumer price index) could signal deflation, which can happen during economic recessions when there is reduced demand for goods and services, leading to lower prices.

How to Handle Negative Values in Our Model?

- No Action Needed if Interpreted Correctly: If these negative values are correctly understood as indicators of economic conditions, they can be used directly in the model to represent their economic meaning.

- Feature Scaling: Depending on the algorithm we're using, it might be beneficial to normalize or standardize these variables to ensure that they have a similar scale to the other features in the dataset. For example, using z-scores or Min-Max scaling can help the model handle both negative and positive values appropriately.

- Contextual Importance: Keep in mind that negative values in these features could indicate important patterns in the data, such as economic conditions influencing customer behavior. Including them can improve the model’s ability to predict client behavior in different economic environments.

**Conclusion:** Negative values in emp.var.rate and cons.price.idx are not inherently problematic. They simply reflect economic conditions such as employment declines or price deflation. As long as we understand their implications, these negative values can be incorporated into the model to help identify patterns and trends related to campaign success.

**Let's have a look at Explortory Data Analysis on categorical data:**

For categorical we will use the subscription status values and review its frquency distribution.
![Frequency Distribution](images/screenshots/img25.jpg)

We started following analysis for each feature :

- **Univariate Analysis:**
>> Univariate analysis involves examining a single variable at a time to understand its distribution and characteristics. It helps in understanding the basic properties of individual features within the dataset.
- **Bivariate Analysis:**
>> Bivariate analysis examines the relationship between two variables. It helps us understand how two variables are related to each other, and if one can help predict the other
- **Multivariate Analysis:**
>> Multivariate analysis involves examining more than two variables at once to understand the complex interactions and relationships among them. Techniques like pair plots, heatmaps of correlations, or 3D scatter plots are useful for visualizing interactions between multiple numerical variables. 



### Splitting the Data into Training and Testing: 

We will begin by splitting the dataset into training and testing subsets. This is a crucial step in building a reliable machine learning model, as it allows us to train the model on one portion of the data and evaluate its performance on unseen data. The split ratio between training and testing sets can be adjusted based on the specific needs of the analysis, though a common practice is to use a standard split such as 70/30 or 80/20. Choosing an appropriate split ensures that the model has enough data to learn meaningful patterns while still being evaluated fairly on data it has not encountered during training.


### Basic Model Development: 

Once the dataset is split into training and testing sets, the next critical step is to build baseline classification models using a variety of algorithms. In this assignment, we focus on four widely used machine learning classifiers: Decision Tree, k-Nearest Neighbors (k-NN), Support Vector Machine (SVM), and Logistic Regression. Each of these models offers unique strengths and approaches to classification tasks. The goal of building these baseline models is to evaluate their initial performance on the dataset, using consistent metrics such as accuracy, precision, recall, and F1-score. This comparison helps identify which algorithm is most suitable for the given problem. Moreover, establishing baseline results sets the foundation for further model tuning and optimization in later stages, such as hyperparameter tuning, feature selection, or ensemble techniques. By systematically evaluating and comparing these models, we can make informed decisions about which classifier best balances performance, complexity, and interpretability for the task at hand. 

### Model Evaluation and Comparision 
Model evaluation in classification tasks is a crucial step to assess how well a model performs in predicting categorical outcomes. It involves using a variety of metrics that go beyond simple accuracy, providing a more comprehensive view of model performance. Common evaluation metrics include the confusion matrix, precision, recall, F1-score, and accuracy. Accuracy measures the overall correctness of the model, but it can be misleading when dealing with imbalanced datasets. Precision focuses on the proportion of correct positive predictions, while recall measures the model's ability to identify all relevant positive cases. The F1-score, as the harmonic mean of precision and recall, offers a balanced evaluation metric, especially useful when false positives and false negatives carry different consequences. Additionally, techniques such as cross-validation help ensure the model's performance is consistent and not just a result of overfitting to a particular dataset. Proper evaluation allows for informed decision-making when selecting and fine-tuning classification models for real-world applications.




