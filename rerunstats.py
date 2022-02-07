# Import Libraries needed:
import matplotlib.pyplot as plt
import pandas
import tools.vicon as vn
import numpy as np
import seaborn as sns
import pingouin as pg

file = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Stats/outputvars-3-1-27-22.xlsx'
savein= 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Python Data Output/Graphs/'
savehere = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Python Data Output/Stat Tables/'

data = pandas.read_excel(file)
dvar = 'trunk_z_totex'

data['Perturbation Intensity'] = data['pertIntensity']
data['Pert. Type'] = data['pert_type']


print(pg.normality(data[dvar]))
pg.qqplot(data[dvar], dist='norm')
plt.show(block=False)
plt.pause(2)
plt.close()
print(pg.sphericity(data=data, dv=dvar, within=['pertType', 'pertlvl'], subject='subject', alpha=0.05))
reslt = pg.rm_anova(dv=dvar, within=['pertType', 'pertlvl'], subject='subject',
                    data=data, detailed=True)
#reslt.to_excel((savehere+'/'+dvar+'_main_effects.xlsx'))
print(reslt.iloc[:, [0, 5, 6, 7]])

# IF interaction effect is significant then compare:
# print(pg.rm_anova(dv=dvar, within=['pertType'], subject='subject', data=lvl1data, detailed=True))
# print(pg.rm_anova(dv=dvar, within=['pertType'], subject='subject', data=lvl2data, detailed=True))
# print(pg.rm_anova(dv=dvar, within=['pertType'], subject='subject', data=lvl3data, detailed=True))
# # if any show that the main effect is stat. significant then we can do 'honest significiant differences'
# # edit data to specify which one was sig.fig. from simplier effects
# print(pg.pairwise_tukey(dv=dvar, data=lvl3data, between='pertType'))


# IF interaction effect is NOT significant, And pertlvl is significant:
nointeract = pg.pairwise_ttests(dv=dvar, data=data, between=['pertType', 'pertlvl'])
#       AND only pertlvl was significant:
reject, pvals_cor = pg.multicomp(nointeract.values[:, 9].tolist(), alpha=0.05, method='bonf')
print(reject, pvals_cor)
print(pg.pairwise_tukey(dv=dvar, data=data, between='pertlvl'))

nointr_bonf = pandas.concat([nointeract.iloc[:, [0, 1, 2, 3, 6, 9]],
                             pandas.DataFrame(data=pvals_cor, columns=('p-bonfcor',))], axis=1)

# if pert type model was significant:
ptyperesults = pg.pairwise_ttests(dv=dvar, data=data, between=['pertlvl', 'pertType'])
print(pg.multicomp(ptyperesults.values[:, 9].tolist(), alpha=0.05, method='bonf'))

#nointr_bonf.to_excel((savehere+'/'+dvar+'.xlsx'))

# Plotting
#for muscles use pallete="husl", otherwise use palette="Paired"
fig, axes = plt.subplots(1, 2)
sns.boxplot(ax=axes[0], data=data, hue="pert_type", y=dvar, x="pertIntensity", palette="Paired")
sns.boxplot(ax=axes[1], data=data, hue="pert_type", y=dvar, x="pertIntensity", palette="Paired")
plt.show()
#fig.savefig((savein + dvar + '.svg'), format='svg')

# fig, axes = plt.subplots(1, 3)
# sns.boxplot(ax=axes[0], data=data, hue="pert_type", y='trunk_x_totex', x="pertIntensity", palette="husl")
# sns.boxplot(ax=axes[1], data=data, hue="pert_type", y='trunk_y_totex', x="pertIntensity", palette="husl")
# sns.boxplot(ax=axes[2], data=data, hue="pert_type", y='trunk_z_totex', x="pertIntensity", palette="husl")
# plt.show()
# fig.savefig((savein + 'trunk.svg'), format='svg')

# gpmn = data.groupby(['pertlvl', 'pertType']).mean()
# gpmn.reset_index(inplace=True)
# musclemean = gpmn.iloc[:, [0,1,19,20,21,22,23,24,25,26,27,28,29,30]]

#
# fig, axes = plt.subplots(1, 2)
# sns.pointplot(ax=axes[0], data=data, hue='Pert. Type', y='trunk_y_totex', x="Perturbation Intensity", palette="husl", ci='sd')
# sns.pointplot(ax=axes[1], data=data, hue='Pert. Type', y='trunk_z_totex', x="Perturbation Intensity", palette="husl", ci='sd')
# plt.show()
# fig.savefig((savein + 'rkneerankle.svg'), format='svg')



fig, axes = plt.subplots(1, 2)
sns.pointplot(ax=axes[0], data=data, hue='Pert. Type', y='tbialis_ant_max', x="Perturbation Intensity",
              palette="Paired", ci='sd')
sns.pointplot(ax=axes[1], data=data, hue='Pert. Type', y='tbialis_ant_iEMG', x="Perturbation Intensity",
              palette="Paired", ci='sd')
plt.show()
fig.savefig((savein + 'TA.svg'), format='svg')

# def pandasPivot(data: pandas.DataFrame, columns2pivot, columns2repeat=None):
#     if columns2repeat is None:
#         matAux = pandas.DataFrame()
#     else:
#         matAux = pandas.concat((data[columns2repeat] for _ in range(len(columns2pivot))))
#
#     matAux['Vals'] = data[columns2pivot].values.T.reshape(data.shape[0] * len(columns2pivot))
#     matAux['Type'] = np.array([[s] * data.shape[0] for s in columns2pivot]).reshape((-1, 1))
#     # matAux['Type'] = data.shape[0] * columns2pivot
#
#     return matAux
#
#
#
# sns.catplot(data=data2, hue="Pert. Type", y='Vals', x="Perturbation Intensity", palette="husl",
#             kind='point', ci='sd', column='Type')