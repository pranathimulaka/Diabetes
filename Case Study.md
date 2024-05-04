# Diabetes
# Introduction
Diabetes is an illness caused because of high glucose level in a human body. Diabetes should not be ignored if it is untreated then Diabetes may cause some major issues in a person like: heart related problems, kidney problem, blood pressure, eye damage and it can also affects other organs of human body. Diabetes is noxious diseases in the world. Diabetes caused because of obesity or high blood glucose level, and so forth. It affects the hormone insulin, resulting in abnormal metabolism of crabs and improves level of sugar in the blood. Diabetes occurs when body does not make enough insulin. Diabetes can be controlled if it is predicted earlier. To achieve this goal this project work we will do early prediction of Diabetes in a human body or a patient for a higher accuracy through applying, Various Machine Learning Techniques. Machine learning techniques Provide better result for prediction by constructing models from datasets collected from patients. In this work we will use Machine Learning Classification and ensemble techniques on a dataset to predict diabetes. Which are K-Nearest Neighbor (KNN), Logistic Regression (LR) and Random Forest (RF). 

# There are primarily four main types of diabetes:

# Type 1 Diabetes (T1D):
Type 1 diabetes is an autoimmune condition where the immune system mistakenly attacks and destroys insulin-producing beta cells in the pancreas.
This results in a complete deficiency of insulin production, requiring lifelong insulin therapy for blood sugar regulation.
Type 1 diabetes often develops in childhood or adolescence, but it can occur at any age.

# Type 2 Diabetes (T2D):
Type 2 diabetes is the most common form of diabetes, characterized by insulin resistance, where the body's cells become less responsive to insulin, and by relative insulin deficiency.
Risk factors for type 2 diabetes include obesity, physical inactivity, poor dietary habits, family history of diabetes, ethnicity, and age.
Type 2 diabetes is often managed with lifestyle modifications (diet, exercise) and may require oral medications or insulin therapy in some cases.

# Gestational Diabetes Mellitus (GDM):
Gestational diabetes occurs during pregnancy and is characterized by elevated blood sugar levels that develop or are first recognized during pregnancy.
While gestational diabetes usually resolves after childbirth, women who have had GDM are at an increased risk of developing type 2 diabetes later in life.
Gestational diabetes is managed with diet, exercise, and, in some cases, insulin therapy to maintain blood sugar levels within a target range during pregnancy.

# Secondary Diabetes:
Secondary diabetes develops as a result of certain medical conditions or medications.
Medical conditions such as pancreatitis, Cushing's syndrome, and hemochromatosis can impair insulin production or action, leading to diabetes.
Some medications, including corticosteroids and certain antipsychotic drugs, can also cause secondary diabetes by affecting blood sugar levels.

# Data Variable Description

A brief description of the variables in the dataset:

Pregnancies :- Number of times a woman has been pregnant

Glucose:-Plasma Glucose concentration of 2 hours in an oral glucose tolerance test

Blood Pressure :- Diastollic Blood Pressure (mm hg)

Skin Thickness:- Triceps skin fold thickness(mm)

Insulin :- 2 hour serum insulin (mu U/ml)

BMI:- Body Mass Index ((weight in kg/height in m)^2)

Age:- Age(years)

Diabetes Pedigree Function :-scores likelihood of diabetes based on family

history

Outcome:- 0(doesn't have diabetes) or 1 (has diabetes)

# OBJECTIVES OF THE PROJECT

The objective of this diabetes dataset is to predict whether patient has diabetes or not. The dataset consists of several medical predictor(independent) variables and one target variable, (outcome). Predictor variables includes pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age and outcome. The accuracy is different for every model when compared to other models. The Project work gives the accurate or higher accuracy model shows that the model is capable of predicting diabetes effectively. Our Result shows that Random Forest achieved higher accuracy compared to other machine learning techniques.

# MACHINE LEARNING TECHNIQUES

# K-Nearest Neighbour (KNN)

K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories. K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm. K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems. K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data. It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset. KNN algorithm at the training phase just stores the dataset and when it gets new data then it classifies as new data.

# Random Forest

Random Forest As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting. Below are some points that explain why we should use the Random Forest algorithm.It takes less training time as compared to other algorithms.It predicts output with high accuracy, even for the large dataset it runs efficiently.It can also maintain accuracy when a large proportion of data is missing.
The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting. Below are some points that explain why weshould use the Random Forest algorithm.It takes less training time as compared to other algorithms.It predicts output with high accuracy, even for the large dataset it runs efficiently.It can also maintain accuracy when a large proportion of data is missing.

# Logistic Regression

Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values (0 or 1). Logistic Regression is a significant machine learning algorithm because it has the ability to provide probabilities and classify new data using continuous and discrete datasets.

# CONCULSION

Case of Study : PREDICTION OF DIABETES USING MACHINE LEARNING ALGORITHMS IN HEALTHCARE,After Modeling the Data i used 3 modeling like :1.Random Forest Classifier,2.Logistic Regression,3.KNeighborsClassifier.I choose the Random Forest Classifier with the Acurracy :0.73828125

# NOVELTY OF THE WORK

Machine learning has the great ability to revolutionize the diabetes risk prediction with the help of advanced computational methods and availability of large amount of epidemiological and genetic diabetes risk dataset. Detection of diabetes in its early stages is the key for treatment. This work has described a machine learning approach to predicting diabetes levels. The technique may also help researchers to develop an accurate and effective tool that will reach at the table of clinicians to help them make better decision about the disease status.

One of the important real-world medical problems is the detection of diabetes at its early stage. In this study, systematic efforts are made in designing a system which results in the prediction of diabetes. During this work, three machine learning classification algorithms are studied and evaluated on various measures. Experiments are performed on Diabetes Database. Experimental results determine the adequacy of the designed system with an achieved accuracy of 73% using Random Forest machine algorithm. In future, the designed system with the used machine learning classification algorithms can be used to predict or diagnose other diseases. The work can be extended and improved for the automation of diabetes analysis including some other machine learning algorithms.
