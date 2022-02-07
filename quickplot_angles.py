## for proposal graph Dec 2021

# Import Libraries needed:
import matplotlib.pyplot as plt
import pandas
import tools.vicon as vn
import numpy as np
import seaborn as sns


filenames = vn.fxn_select_files()  # Ask user to select baseline joint angles
stringname = ''.join(filenames[0])
subject = stringname[stringname.index("sub"):stringname.index("sub") + 5]  # string
subject_index = int(subject[3:]) - 1  # subject_index = python index of subject, integer value
print(subject)

for i in range(0, len(filenames)):
    if i == 0:
        # for angles its read_csv - updated for gnd forces
        data = pandas.read_excel(filenames[i])
    else:
        data = pandas.concat([data, pandas.read_excel(filenames[i])])

data.reset_index(inplace=True)
g = sns.relplot(data=data, x="Cycle", y="avg_flex", hue="Joint", col="subject", kind="line")
g.map(sns.kdeplot)
plt.show()
# trying line plot
sns.relplot(data=data, x="Cycle", y="avg_flex", col="Joint")
sns.relplot(data=data, x="Cycle", y="avg_flex", hue="Joint")
data.reset_index(inplace=True)
sns.relplot(data=data, x="Cycle", y="avg_flex", col="Joint", kind="line")
sns.relplot(data=data, x="Cycle", y="avg_flex", hue="Joint", kind="line")
sns.relplot(data=data, x="Cycle", y="avg_flex",  hue="Joint", style="subject")




