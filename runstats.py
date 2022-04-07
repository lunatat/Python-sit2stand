# After running step01_load.py - compile subject data into 1 csv and upload file to run some stats!

# Affiliation : ROAR Lab
# Project: TPAD sit-to-stand: study01
# Author: Tatiana D. Luna
# Date: 1.24.22

# Import Libraries needed:
import matplotlib.pyplot as plt
import pandas
import tools.vicon as vn
import numpy as np
import seaborn as sns
import pingouin as pg

savehere = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Python Data Output/Stat Tables/'
savein= 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Python Data Output/Graphs/'
#file = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Stats/outputvars-3pt1-1-24-22.xlsx'
#file = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Stats/outputvars-3pt1-1-24-22-testingwonly2trails.xlsx'
file = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Stats/outputvars-3-1-27-22.xlsx'
#data = pandas.read_csv(file)
#data = pandas.read_excel(file)

filenames = vn.fxn_select_files()  # Ask user to select cvs files to load
for i in range(0, len(filenames)):
    if i == 0:
        data = pandas.read_excel(filenames[i])
    else:
        data = pandas.concat([data, pandas.read_excel(filenames[i])])
data.reset_index(inplace=True)
#
# #data = pandas.read_excel(filenames[0])

dvar = 'bicep_fem_max'

# # normality test: default is shapiro
# print(pg.normality(data[dvar]))
# # Q-Q plots
# pg.qqplot(data[dvar], dist='norm')
# plt.show()
# # dv-dependept var
# res = pg.rm_anova(dv=dvar, within=['pertType', 'pertlvl'], subject='subject', data=data, detailed=True)
# # boxplots:
# sns.boxplot(data=data, x="pertlvl", y=dvar, hue="pertType")
#
# lvl1 stats: look at variable only during first perturbation
# lvl1data = data.loc[data['pertlvl'] == 1]
# print(pg.normality(lvl1data[dvar]))
# pg.qqplot(lvl1data[dvar], dist='norm')
# plt.show(block=False)
# plt.pause(1)
# plt.close()
# print(pg.sphericity(data=lvl1data, dv=dvar, within=['pertType', 'pertDirt'], subject='subject', alpha=0.05))
# resltlvl1 = pg.rm_anova(dv=dvar, within=['pertType', 'pertDirt'], subject='subject',
#                    data=lvl1data, detailed=True)
# print(resltlvl1.iloc[:, [0,5,6,7]])
# sns.boxplot(data=lvl1data, hue="pertDirt", y=dvar, x="pertType")
# plt.show()
#
# # lvl2 stats: look at variable only during threshold perturbation
# lvl2data = data.loc[data['pertlvl'] == 2]
# print(pg.normality(lvl2data[dvar]))
# pg.qqplot(lvl2data[dvar], dist='norm')
# plt.show(block=False)
# plt.pause(1)
# plt.close()
# print(pg.sphericity(data=lvl2data, dv=dvar, within=['pertType', 'pertDirt'], subject='subject', alpha=0.05))
# resltlvl2 = pg.rm_anova(dv=dvar, within=['pertType', 'pertDirt'], subject='subject',
#                    data=lvl2data, detailed=True)
# print(resltlvl2.iloc[:, [0,5,6,7]])
# sns.boxplot(data=lvl2data, hue="pertDirt", y=dvar, x="pertType")
# plt.show()
# #
# # lvl3 stats: look at variable only during failing pt
# lvl3data = data.loc[data['pertlvl'] == 3]
# print(pg.normality(lvl3data[dvar]))
# pg.qqplot(lvl3data[dvar], dist='norm')
# plt.show(block=False)
# plt.pause(1)
# plt.close()
# print(pg.sphericity(data=lvl3data, dv=dvar, within=['pertType', 'pertDirt'], subject='subject', alpha=0.05))
# resltlvl3 = pg.rm_anova(dv=dvar, within=['pertType', 'pertDirt'], subject='subject',
#                    data=lvl3data, detailed=True)
# print(resltlvl3.iloc[:, [0,5,6,7]])
# sns.boxplot(data=lvl3data, hue="pertDirt", y=dvar, x="pertType")
# plt.show()

# look at variable only during 1-3 **************************************************
datawout4 = data.loc[data['pertlvl'].isin([1, 2, 3])]
bins = [0, 1, 2, 3]
pnames = ['First', 'Threshold', 'Fail']
datawout4['pertIntensity'] = pandas.cut(datawout4['pertlvl'], bins, labels=pnames)
bins = [-1, 0, 1]
tnames = ['Pelvis', 'Treadmill']
datawout4['pert_type'] = pandas.cut(datawout4['pertType'], bins, labels=tnames)
# datawout4['SI_max'] = datawout4['LFznorm_max'] - datawout4['RFznorm_max']

print(pg.normality(datawout4[dvar]))
pg.qqplot(datawout4[dvar], dist='norm')
plt.show(block=False)
plt.pause(1)
plt.close()
print(pg.sphericity(data=datawout4, dv=dvar, within=['pertType', 'pertlvl'], subject='subject', alpha=0.05))
resltwout4 = pg.rm_anova(dv=dvar, within=['pertType', 'pertlvl'], subject='subject',
                         data=datawout4, detailed=True)
#reslt.to_excel((savehere+'/'+dvar+'_main_effects.xlsx'))
print(resltwout4.iloc[:, [0, 5, 6, 7]])

# IF interaction effect is significant then compare:
# print(pg.rm_anova(dv=dvar, within=['pertType'], subject='subject', data=lvl1data, detailed=True))
# print(pg.rm_anova(dv=dvar, within=['pertType'], subject='subject', data=lvl2data, detailed=True))
# print(pg.rm_anova(dv=dvar, within=['pertType'], subject='subject', data=lvl3data, detailed=True))
# # if any show that the main effect is stat. significant then we can do 'honest significiant differences'
# # edit data to specify which one was sig.fig. from simplier effects
# print(pg.pairwise_tukey(dv=dvar, data=lvl3data, between='pertType'))


# IF interaction effect is NOT significant, And pertlvl is significant:
nointeract = pg.pairwise_ttests(dv=dvar, data=datawout4, between=['pertType', 'pertlvl'])
#       AND only pertlvl was significant:
reject, pvals_cor = pg.multicomp(nointeract.values[:, 9].tolist(), alpha=0.05, method='bonf')
print(reject, pvals_cor)
print(pg.pairwise_tukey(dv=dvar, data=datawout4, between='pertlvl'))

nointr_bonf = pandas.concat([nointeract.iloc[:, [0, 1, 2, 3, 6, 9]],
                             pandas.DataFrame(data=pvals_cor, columns=('p-bonfcor',))], axis=1)

# if pert type model was significant:
ptyperesults = pg.pairwise_ttests(dv=dvar, data=datawout4, between=['pertlvl', 'pertType'])

#nointr_bonf.to_excel((savehere+'/'+dvar+'.xlsx'))

# Plotting
#for muscles use pallete="husl", otherwise use palette="Paired"
fig, axes = plt.subplots(1, 2)
sns.boxplot(ax=axes[0], data=datawout4, hue="pert_type", y=dvar, x="pertIntensity", palette="husl")
sns.boxplot(ax=axes[1], data=datawout4, hue="pert_type", y=dvar, x="pertIntensity", palette="husl")
plt.show()
#fig.savefig((savein + dvar + '_knee.svg'), format='svg')

# # comparing against baseline
# datawbaseline = data.loc[data['pertlvl'].isin([0, 1, 2, 3])]
# resultwbaseline = pg.rm_anova(dv=dvar, within=['pertlvl'], subject='subject',
#                          data=datawbaseline, detailed=True)
# print(resultwbaseline.iloc[:, [0, 5, 6, 7]])
# pg.pairwise_ttests(dv=dvar, data=datawbaseline, between=['pertlvl'])