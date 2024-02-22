import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from sklearn.model_selection import cross_val_score, RepeatedKFold, RepeatedStratifiedKFold



methods = ['Oringin','Univariate ANOVA', 'Mutual Information', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'L1 Regularization']

data_frames = {}

for method in methods:
    file_name = f"feature_selection_{method.replace(' ', '_')}_results.csv"
    data_frames[method] = pd.read_csv(file_name)

univariate_anova_df = data_frames['Univariate ANOVA']
mutual_information_df = data_frames['Mutual Information']
random_forest_df = data_frames['Random Forest']
gradient_boosting_df = data_frames['Gradient Boosting']
xgboost_df = data_frames['XGBoost']
l1_regularization_df = data_frames['L1 Regularization']
all_species_df = data_frames['Oringin']

labs=[]
for i in range(len(mutual_information_df)):
    labs.append(mutual_information_df.iloc[i,1])
labs=pd.DataFrame(labs)
labs

lab = labs
enc = LabelEncoder()
labe = enc.fit_transform(lab.values)
labe

X = np.array(mutual_information_df.iloc[:,2:])
y = np.array(labs)
X

models = [KNeighborsClassifier(), SVC(probability=True,random_state=0), DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(random_state =0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0), XGBClassifier(use_label_encoder=False, n_jobs=1,random_state =0), 
          XGBRFClassifier(use_label_encoder=False, n_jobs=1,random_state =0),LGBMClassifier(random_state =0)]
names = ['KNN', 'SVM', 'DT', 'RF', 'GB', 'XGB', 'XGBRF', 'LGB']

cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

aucs = []
for i,n in zip(names, models):
    print(i)
    auc = cross_val_score(n, mutual_information_df.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    aucs.append(auc)


pd.DataFrame(aucs, index=names)


#random_forest
labs=[]
for i in range(len(random_forest_df)):
    labs.append(random_forest_df.iloc[i,1])
labs=pd.DataFrame(labs)
labs

lab = labs
enc = LabelEncoder()
labe = enc.fit_transform(lab.values)
labe

X = np.array(random_forest_df.iloc[:,2:])
y = np.array(labs)
X

models = [KNeighborsClassifier(), SVC(probability=True,random_state=0), DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(random_state =0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0), XGBClassifier(use_label_encoder=False, n_jobs=1,random_state =0), 
          XGBRFClassifier(use_label_encoder=False, n_jobs=1,random_state =0),LGBMClassifier(random_state =0)]
names = ['KNN', 'SVM', 'DT', 'RF', 'GB', 'XGB', 'XGBRF', 'LGB']

cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

aucs = []
for i,n in zip(names, models):
    print(i)
    auc = cross_val_score(n, random_forest_df.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    aucs.append(auc)


pd.DataFrame(aucs, index=names)


#gradient_boosting
labs=[]
for i in range(len(gradient_boosting_df)):
    labs.append(gradient_boosting_df.iloc[i,1])
labs=pd.DataFrame(labs)
labs

lab = labs
enc = LabelEncoder()
labe = enc.fit_transform(lab.values)
labe

X = np.array(gradient_boosting_df.iloc[:,2:])
y = np.array(labs)
X

models = [KNeighborsClassifier(), SVC(probability=True,random_state=0), DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(random_state =0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0), XGBClassifier(use_label_encoder=False, n_jobs=1,random_state =0), 
          XGBRFClassifier(use_label_encoder=False, n_jobs=1,random_state =0),LGBMClassifier(random_state =0)]
names = ['KNN', 'SVM', 'DT', 'RF', 'GB', 'XGB', 'XGBRF', 'LGB']
#cv = StratifiedKFold(n_splits=4)
cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

aucs = []
for i,n in zip(names, models):
    print(i)
    auc = cross_val_score(n, gradient_boosting_df.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    aucs.append(auc)


pd.DataFrame(aucs, index=names)


#xgboost
labs=[]
for i in range(len(xgboost_df)):
    labs.append(xgboost_df.iloc[i,1])
labs=pd.DataFrame(labs)
labs

lab = labs
enc = LabelEncoder()
labe = enc.fit_transform(lab.values)
labe

X = np.array(xgboost_df.iloc[:,2:])
y = np.array(labs)
X

models = [KNeighborsClassifier(), SVC(probability=True,random_state=0), DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(random_state =0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0), XGBClassifier(use_label_encoder=False, n_jobs=1,random_state =0), 
          XGBRFClassifier(use_label_encoder=False, n_jobs=1,random_state =0),LGBMClassifier(random_state =0)]
names = ['KNN', 'SVM', 'DT', 'RF', 'GB', 'XGB', 'XGBRF', 'LGB']

cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

aucs = []
for i,n in zip(names, models):
    print(i)
    auc = cross_val_score(n, xgboost_df.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    aucs.append(auc)


pd.DataFrame(aucs, index=names)


#l1_regularization
labs=[]
for i in range(len(l1_regularization_df)):
    labs.append(l1_regularization_df.iloc[i,1])
labs=pd.DataFrame(labs)
labs

lab = labs
enc = LabelEncoder()
labe = enc.fit_transform(lab.values)
labe

X = np.array(l1_regularization_df.iloc[:,2:])
y = np.array(labs)
X

models = [KNeighborsClassifier(), SVC(probability=True,random_state=0), DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(random_state =0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0), XGBClassifier(use_label_encoder=False, n_jobs=1,random_state =0), 
          XGBRFClassifier(use_label_encoder=False, n_jobs=1,random_state =0),LGBMClassifier(random_state =0)]
names = ['KNN', 'SVM', 'DT', 'RF', 'GB', 'XGB', 'XGBRF', 'LGB']
#cv = StratifiedKFold(n_splits=4)
cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

aucs = []
for i,n in zip(names, models):
    print(i)
    auc = cross_val_score(n, l1_regularization_df.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    aucs.append(auc)


pd.DataFrame(aucs, index=names)



dataframes = [all_species_df , univariate_anova_df , mutual_information_df , gradient_boosting_df, xgboost_df , random_forest_df , gradient_boosting_df , l1_regularization_df]
df_names = ['all_species_df' , 'univariate_anova_df','mutual_information_df' , 'gradient_boosting_df', 'xgboost_df' , 'random_forest_df' , 'gradient_boosting_df' , 'l1_regularization_df']

results = {}

for df, name in zip(dataframes, df_names):
    labs = []
    for i in range(len(df)):
        labs.append(df.iloc[i, 1])
    labs = pd.DataFrame(labs)

    lab = labs
    enc = LabelEncoder()
    labe = enc.fit_transform(lab.values)

    X = np.array(df.iloc[:, 2:])
    y = np.array(labs)

    models = [KNeighborsClassifier(), SVC(probability=True, random_state=0), DecisionTreeClassifier(random_state=0),
              RandomForestClassifier(random_state=0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), LGBMClassifier(random_state =0),XGBClassifier(use_label_encoder=False, n_jobs=1, random_state=0), 
            XGBRFClassifier(use_label_encoder=False, n_jobs=1, random_state=0)]
    names = ['KNN', 'SVM', 'DT', 'RF', 'GB','LGB','XGB', 'XGBRF' ]

    cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

    aucs = []
    for i, n in zip(names, models):
        print(f"{name} - {i}")
        auc = cross_val_score(n, df.iloc[:, 2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        aucs.append(auc)

    results[name] = pd.Series(aucs, index=names)

combined_results = pd.DataFrame(results)

csv_path = "combined_results.csv"
combined_results.to_csv(csv_path)

print(combined_results)


