import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy as np

file = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Stats/outputs-4-7-22-(2).xlsx'
widefile = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Stats/outputs-4-7-22-for3RM.xlsx'
savein= 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Python Data Output/Graphs/'

data = pandas.read_excel(file)
dataW = pandas.read_excel(widefile)

data['Direction'] = data['pertDirt']
data.loc[data['Direction'] == 0, ['Direction']] = 'Post.'
data.loc[data['Direction'] == 4, ['Direction']] = 'Ant.'

spssdata = data.loc[data['subject'].isin([1, 3, 5, 6, 7, 9, 12, 13, 14, 15])]

dvar = 'medial_gast_iEMG'

fig, axes = plt.subplots(1, 2, sharey=True)
sns.pointplot(ax=axes[0], data=spssdata.loc[spssdata['Pert. Type'] == 'Pelvis'], hue='Direction', y=dvar,
              x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], dodge=True,
              palette="Paired", ci='sd', capsize=0.2)
s1 = sns.swarmplot(ax=axes[0], data=spssdata.loc[spssdata['Pert. Type'] == 'Pelvis'], hue='Direction', y=dvar,
                   dodge=True,
                   x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], palette="Paired")
handles, labels = s1.get_legend_handles_labels()
s1.legend(handles[:2], labels[:2])
axes[0].title.set_text('Pelvis')
sns.pointplot(ax=axes[1], data=spssdata.loc[spssdata['Pert. Type'] == 'Treadmill'], hue='Direction', y=dvar,
              x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], dodge=True,
              palette="Paired", ci='sd', capsize=0.2)
s2 = sns.swarmplot(ax=axes[1], data=spssdata.loc[spssdata['Pert. Type'] == 'Treadmill'], hue='Direction', y=dvar,
                   dodge=True,
                   x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], palette="Paired")
handles, labels = s2.get_legend_handles_labels()
s2.legend(handles[:2], labels[:2])
axes[1].title.set_text('Treadmill')
plt.show()
fig.savefig((savein + dvar + '.svg'), format='svg')

# save kinematic variables:
dvar = 'trunk_z_totex'

p = sns.color_palette("Paired", 10)
otherpair = sns.color_palette([p[8], p[9], p[6], p[7]])

fig, axes = plt.subplots(1, 2)
sns.pointplot(ax=axes[0], data=spssdata, hue='Direction', y=dvar,
              x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], dodge=True,
              palette=otherpair, ci='sd', capsize=0.2)
s1 = sns.swarmplot(ax=axes[0], data=spssdata, hue='Direction', y=dvar,
                   dodge=True,
                   x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], palette=otherpair)
handles, labels = s1.get_legend_handles_labels()
s1.legend(handles[:2], labels[:2])

sns.pointplot(ax=axes[1], data=spssdata, hue='Direction', y='trunk_z_totex',
              x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], dodge=True,
              palette=otherpair, ci='sd', capsize=0.2)
s2 = sns.swarmplot(ax=axes[1], data=spssdata, hue='Direction', y='trunk_z_totex',
                   dodge=True,
                   x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], palette=otherpair)
handles, labels = s2.get_legend_handles_labels()
s2.legend(handles[:2], labels[:2])
plt.show()
fig.savefig((savein + dvar + '_andZ.svg'), format='svg')

fig, axes = plt.subplots(1, 2, sharey=True)
sns.pointplot(ax=axes[0], data=spssdata.loc[spssdata['Pert. Type'] == 'Pelvis'], hue='Direction', y=dvar,
              x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], dodge=True,
              palette=otherpair, ci='sd', capsize=0.2)
s1 = sns.swarmplot(ax=axes[0], data=spssdata.loc[spssdata['Pert. Type'] == 'Pelvis'], hue='Direction', y=dvar,
                   dodge=True,
                   x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], palette=otherpair)
handles, labels = s1.get_legend_handles_labels()
s1.legend(handles[:2], labels[:2])
axes[0].title.set_text('Pelvis')
sns.pointplot(ax=axes[1], data=spssdata.loc[spssdata['Pert. Type'] == 'Treadmill'], hue='Direction', y=dvar,
              x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], dodge=True,
              palette=otherpair, ci='sd', capsize=0.2)
s2 = sns.swarmplot(ax=axes[1], data=spssdata.loc[spssdata['Pert. Type'] == 'Treadmill'], hue='Direction', y=dvar,
                   dodge=True,
                   x="Perturbation Intensity", hue_order=['Ant.', 'Post.'], palette=otherpair)
handles, labels = s2.get_legend_handles_labels()
s2.legend(handles[:2], labels[:2])
axes[1].title.set_text('Treadmill')
plt.show()
fig.savefig((savein + dvar + '.svg'), format='svg')

# just group average left knee
dvar = 'r_knee_x_totex'
sns.pointplot(data=spssdata, y=dvar, x='Direction', palette=otherpair, ci='sd', capsize=0.2)
sns.swarmplot(data=spssdata, y=dvar, x='Direction', palette=otherpair)
