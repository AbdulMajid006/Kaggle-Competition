# Kaggle-Competition
This repo contains code and submissions of kaggle competition


Kaggle Competition Report
Name: Abdul Majid
Roll No: 24k-0895
Section: 4H
Submission Date: 26-Apr-2026
















1.	Competition Overview:
Competition Name: Playground series
Problem Type: Multi-Class Classification
Evaluation Metric: Accuracy (macro-averaged)
Brief Description of the Dataset: 
The dataset contains agricultural and environmental sensor readings for farm fields. Each row represents a field observation with features including Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity, Temperature_C,Humidity, Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh, Crop_Type, Crop_Growth_Stage, Season, Irrigation_Type, Water_Source, Field_Area_hectare, Mulching_Used, Previous_Irrigation_mm, and Region. The goal is to predict the irrigation need level (Low, Medium, or High) for each field. The dataset is imbalanced — Low class dominates with 58.7% of samples while High class is a minority at only 3.3%.
2.	Data Processing:
Missing values handling: 
Missing values were minimal across all columns. Numerical columns were filled with the
median of the training set. Categorical columns were filled with the mode. The same fill
values were applied to the test set to prevent data leakage.
Encoding Techniques: 
The target column Irrigation_Need was Label Encoded: High=0, Low=1, Medium=2. All
categorical features (Soil_Type, Crop_Type, Season, Water_Source, Region,
Mulching_Used, Crop_Growth_Stage, Irrigation_Type) were Label Encoded per column.
Feature Scaling: 
StandardScaler was applied for distance-based and linear models: KNN, Logistic Regression, Naive Bayes, and K-Means. Tree-based models (Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM) used raw unscaled features.
Train-Validation Techniques: 
An 80/20 stratified split was used — 504,000 training rows and 126,000 validation rows.
Stratification preserved the class distribution across both sets. Slow models (K-Means, KNN, Gradient Boosting) trained on a 50,000






3.	Models Attempted:
Decision Tree: 
Two variants were tested — a fully grown baseline tree and a pruned tuned version.
 
 













Naive Bayes: 
Gaussian Naive Bayes was used on scaled features. Assumes all features are independent — a poor assumption for correlated agricultural data.
 











K-Means as Classifier: 
KMeans was fitted with n_clusters = number of classes (3). Each cluster was assigned the majority class label of the training samples that fell into it.
 











Logistic Regression: 
Logistic Regression with saga solver and 500 iterations on scaled features.
 











Advanced Models: 
Three ensemble models were tested for higher accuracy.
 

 
 
 

 
 
4.	Cross Validation Strategy:
K-Fold Details: 
Stratified 3-Fold Cross-Validation was applied to all major models. K was set to 3 instead of the standard 5 due to the large dataset size (630k rows). Stratification ensures each fold has proportional class representation.
Model	3-Fold CV Accuracy
Decision Tree (Tuned)	0.9783
Naive Bayes	0.7123
Logistic Regression	0.7577
Random Forest	0.9840
XGBoost Tuned	0.9846
LightGBM Tuned	0.9828

LOOCV Observations:
Leave-One-Out Cross-Validation was performed on a 200-sample random subset using a Decision Tree (max_depth=10). Full LOOCV on 630k rows is computationally infeasible as it requires 630,000 individual model fits. LOOCV Result on 200-sample subset: 0.7300 ± 0.4440. The high standard deviation is expected with LOOCV as each test set is a single sample — variance is inherently high. The mean confirms the Decision Tree generalises reasonably on unseen data.
Best Validation Accuracy:
Best validation accuracy achieved: 0.9855 by Random Forest, XGBoost Tuned, and LightGBM Tuned (all three tied on the holdout set).





5.	Failed Attempts and Insights:
Decision Tree:
Model Used: Decision Tree
Accuracy Obtained: 0.9695
What went wrong? A fully grown tree with no depth limit memorises the training data. It creates leaves for individual samples leading to high training accuracy but reduced
generalizsation.
What was improved? Setting max_depth=15 and min_samples_leaf=5 improved val accuracy from 0.9695 to 0.9801. Pruning forces the model to learn general patterns rather than memorise noise.
 





K-Means Classifier:
Model Used: K-Means Classifier
Accuracy Obtained: 0.6632
What went wrong? KMeans is unsupervised — it finds geometric clusters in feature space without using labels. These clusters do not align with irrigation need boundaries. The High class was completely missed (precision=0.00) because the rare High samples were absorbed into Low/Medium clusters.
What was improved? Confirmed that supervised methods are necessary for this problem. Switching to any supervised model immediately improved accuracy. This experiment demonstrated the value of labelled training data.
 




Naive Bayes:
Model Used: Naive Bayes
Accuracy Obtained: 0.7137
What went wrong? Gaussian Naive Bayes assumes all features are statistically independent. Agricultural features like Soil_Moisture, Rainfall, Temperature, and Humidity are strongly correlated — violating this assumption. The model struggled
particularly with the Medium class (recall=0.34).
What was improved? Using tree-based models that inherently handle feature correlations improved accuracy dramatically. Feature engineering also helped by creating explicit interaction terms.
 





Logistic Regression:
Model Used: Logistic Reggression
Accuracy Obtained: 0.7585
What went wrong? Linear decision boundaries cannot capture non-linear relationships between environmental features and irrigation need. The convergence warning (max_iter reached) shows the model struggled to find an optimal solution even with 500 iterations. High class recall was only 0.38.
What was improved? Non-linear ensemble models solved both issues. Feature engineering created nonlinear interaction terms that partially helped linear models but ensemble methods still outperformed significantly.
 




6.	Final Model Selection:
Best Model: 
Random Forest achieved the highest validation accuracy (0.9855) and best K-Fold CV score (0.9840 ± 0.0006) among all models. It was retrained on the full 630,000 training rows before generating test predictions.
Hyperparameters: 
Hyperparameter	Value	Reason
n_estimators	300	More trees = better generalisation
Max_depth	20	Deep enough to capture complexity without full
overfit
n_jobs	-1	Use all CPU cores for parallel training
random_state	42	Reproducibility

Why Random Forest Selected: 
•	Highest validation accuracy: 0.9855 (tied with XGBoost and LightGBM but with better CV score)
•	Best K-Fold CV: 0.9840 ± 0.0006 — lowest variance confirming stable generalisation
•	Strong performance on minority High class: precision=0.97, recall=0.91
•	Ensemble of 200-300 decision trees reduces variance compared to a single tree
•	No feature scaling required — robust to outliers and different feature scales
•	Feature importance available — Crop_Growth_Stage (29.5%) and Soil_Moisture (14.4%) top features
 
 
7.	Leaderboard Performance:
Kaggle Score: 0.96752
Rank: 1436
 
8.	Conclusion and Learnings:
Key Insights: 
•	Tree-based ensemble methods (Random Forest, XGBoost, LightGBM) all achieved ~98.5% accuracy — far above linear and probabilistic models
•	Feature engineering was highly impactful: 13 new interaction features increased the feature count from 19 to 32 and improved model accuracy
•	Crop_Growth_Stage was the most important feature (29.5%) — indicating plant growth stage is the strongest predictor of irrigation need
•	Soil_Moisture (14.4%) and Mulching_Used (10.6%) were the next most important, confirming domain logic
•	K-Means completely failed on the minority High class — unsupervised methods cannot replace supervised classification on labelled data
•	Naive Bayes and Logistic Regression performed similarly (~71-76%) as both are limited by linearity assumptions
•	The dataset is imbalanced (High class = 3.3%) — models need to handle this; Random Forest did so naturally through ensemble voting


Challenges Faced: 
•	Dataset size (630k rows) made full LOOCV and 5-fold CV on slow models (Gradient Boosting, KNN) infeasible — required 50k sampling
•	Gradient Boosting took 244 seconds even on 50k rows; full training would take several hours
•	Target column name was Irrigation_Need not irrigation_type — required correcting before code ran
•	Kaggle file path was /kaggle/input/competitions/playground-series-s6e4/ not
•	/kaggle/input/playground-series-s6e4/
•	Save Version on Kaggle took 40+ minutes to re-execute the full pipeline on 630k rows
•	Logistic Regression did not converge in 500 iterations on this large dataset
Future Improvements: 
•	Hyperparameter tuning with Optuna for automated search — could push Random Forest accuracy higher
•	Stacking/Blending: combine Random Forest + XGBoost + LightGBM predictions for a small accuracy gain
•	CatBoost: designed for categorical features, may outperform on this dataset with many categorical columns
•	Handle class imbalance explicitly using SMOTE oversampling or class_weight='balanced'
•	Target encoding for high-cardinality categorical features instead of label encoding
•	Neural network TabNet — designed for tabular data and may compete with boosting methods

