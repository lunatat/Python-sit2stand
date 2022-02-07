# Import Libraries needed:
import matplotlib.pyplot as plt
import pandas
import tools.vicon as vn
import numpy as np
import seaborn as sns
import pingouin as pg

file = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Stats/outputvars-3pt1-1-24-22-testingwonly2trails.xlsx'
#data = pandas.read_csv(file)
data = pandas.read_excel(file)
subfile = 'D:/GitHub Content/RobUST/RobUST/TPAD-subjects.xlsx'
thresholds = pandas.read_excel(subfile, sheet_name='Thresholds')

# for subjects: 4, 5, 8, 9, 10, 11 their first instance was their threshold
# 1. loop through and copy first instance as threshold
# 2. delete any additional pert lvl 3's besides the first 3 instance
# 3. delete everything else
# 4. save data
badsubjects = np.array([4, 5, 8, 9, 10, 11]) - 1
dirt = [4, 0, 4, 0]
minval = [5, 0.4]
# loop by through subject
for i in range(0, len(thresholds)):

    subdata = data.loc[data['subject'] == i]
    lvl1 = subdata.eq([0, 5, 1], index=["pertType", "pertValue", "pertlvl"])

    for k in [0, 1, 2, 3]:
        # loop through perturbation combinations
        findthis = thresholds.values[i, k+1]
        # find first pertlvl 1 instance for the pert combo and keep that
        valuesindirt = subdata.loc[subdata['pertDirt'] == dirt[k]]  # all data in that direction
        #next(one for one in valuesindirt.values.tolist() if one == 1)
        lvl1s = valuesindirt.loc[subdata['pertlvl'] == 1]
        if len(lvl1s) > 2:
            # get only first instance of first type of perturbation

            lvl1s['pertType'].values.tolist()
        keepthis = pandas.concat([keepthis, lvl1s])
        # find subjects that had first instance also be threshold:
        if i in badsubjects:
            if findthis in minval:
                lvl2 = subdata.loc[subdata['pertValue'] == findthis]
                lvl2 = lvl2.loc[lvl2['pertDirt'] == dirt[k]]
                lvl2.at[0, 'pertlvl'] = 2
            keepthis = pandas.concat([keepthis, lvl2])
        # find lvl2s
        lvl2s = subdata.loc[subdata['pertlvl'] == 2]
        keepthis = pandas.concat([keepthis, lvl2s])
        # find any extra 3's
        lvl2s = subdata.loc[subdata['pertlvl'] == 2]