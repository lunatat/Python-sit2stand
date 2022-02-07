import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

## Plotting the threshold data: *****************************************************************
file = 'D:\GitHub Content\RobUST\RobUST\TPAD-subjects.xlsx' # make sure to update file location!
thresholds = pandas.read_excel(file, sheet_name='Thresholds')
sns.set_theme(style="whitegrid") # gives it that pastel color
ax = sns.barplot(data=thresholds[["Pelvis N", "Pelvis S"]]) # plots mean+std
ax.set(xlabel="Perturbation Type", ylabel="Threshold Averages")
plt.show()

ax = sns.barplot(data=thresholds[["Treadmill N", "Treadmill S"]])
ax.set(xlabel="Perturbation Type", ylabel="Threshold Averages")
plt.show()

