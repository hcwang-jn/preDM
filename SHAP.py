#SHAP
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

color1 = (1, 1, 1)      
color2 = (46/255, 120/255, 175/255)  
colors = [color1, color2]
n_bins = 100  
cmap_name = "my_custom_cmap"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

plt.figure(figsize=(10, 8))
sns.heatmap(combined_results, annot=True, cmap=custom_cmap, fmt=".3f", linewidths=0.5, annot_kws={"size": 12})


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import shap
import matplotlib.pyplot as plt

otu_table = pd.read_csv('feature_selection.csv', index_col=0)
labels = pd.read_csv('map_list.csv', index_col=0)

X = otu_table.iloc[:, 1:]  
labels1 = LabelEncoder().fit_transform(labels['Group'])  

rf_classifier =GradientBoostingClassifier()
rf_classifier.fit(X, labels1)

explainer = shap.KernelExplainer(rf_classifier.predict_proba, X)

index = 0  
shap_values_single = explainer.shap_values(X.iloc[index])

shap_values_all = explainer.shap_values(X)

print("SHAP values for single instance:")
print(shap_values_single)

feature_index = 0 
print(f"SHAP values for feature {feature_index}:")
print(shap_values_all[feature_index])

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values_single[1], X.iloc[index])

shap.summary_plot(shap_values_all[1], X)

fig = plt.figure()

shap.summary_plot(shap_values_all[1], X, show=False)

fig.savefig("species_cd_shap_plot.pdf", dpi=300, bbox_inches="tight")

plt.show()