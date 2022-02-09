# Loads raw mrk, fplate, emg, pxi -> compile raw data based on perturbation data -> export pickle
# user will only be asked to select marker data
# compiling data to use for prediction/RNN

# Affiliation : ROAR Lab
# Project: TPAD sit-to-stand: study01
# Author: Tatiana D. Luna
# Date: 2.7.22

import pandas
import tools.vicon as vn
import pickle as pkl
import tools.analyze as lyz
import numpy as np

# File Paths from DESKTOP:
emgpath = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/EMG Export/'
fppath = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Force Plate Export/'
pxipath = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/PXI/'
subfile = 'D:/GitHub Content/RobUST/RobUST/TPAD-subjects.xlsx'  # delete?

filenames = vn.fxn_select_files()  # Ask user to select raw marker files to load
stringname = ''.join(filenames[0])
subject = stringname[stringname.index("sub"):stringname.index("sub") + 5]  # string
subject_index = int(subject[3:]) - 1  # subject_index = python index of subject, integer value
thresholds = pandas.read_excel(subfile, sheet_name='Thresholds')
mrkpath = stringname[0:stringname.index("sub") + 6]
print(subject)

if subject_index == 0:
    fplatecolumnind = [2, 11]
else:
    fplatecolumnind = [2, 17]

fplatenames = ('LFz', 'RFz', 'LFznorm', 'RFznorm')

# open files selected and the associated EMG, Forceplate and PXI file *********************
for i in range(0, len(filenames)):
    stringname = ''.join(filenames[i])  # converts to string
    file = stringname[stringname.index("sub") + 6:-4]  # find index of filename only(after sub before .csv)
    print('Loading MRK: ', file)
    mrkdata = vn.read_vicon_XYZ_file(filenames[i])
    time_sets = lyz.splitsit2stand(mrkdata, subject_index)  # time (s) of when sub starts&finishes standing
    mt = np.arange(0, len(mrkdata.values[:, 0]) / 200, 1 / 200)  # mrk time
    print('Loading EMG: ', file)
    emg = vn.read_emg_csv(vn.getpath(emgpath, subject, file))
    et = np.arange(0, len(emg.values[:, 0]) / 2000, 1 / 2000)  # emg time
    # 1. label everything zero @ first = stable, no perturbation
    emg['stable'] = 0
    mrkdata['stable'] = 0

    if file[0:4] != 'pert':  # load baseline:
        pert = pandas.DataFrame(data=[[-1, -1, -1, 0, -1, 0, 0, 0]],
                                columns=('pertStart', 'pertEnd', 'pertType', 'pertValue', 'pertDirt',
                                         'pertampVal', 'pertlvl', 'heightwhenpert'))
    else:  # load pert files
        print('Loading PXI: ', file)
        pxi = vn.read_pxi_txt(pxipath + subject + '/' + subject + '_' + file[0:4] + '_' + file[4:6] + '.txt')
        # pert is dataframe of ('pertStart', 'pertEnd', 'pertType', 'pertValue', 'pertDirt',
        # 'pertampVal', 'pertlvl', 'heightwhenpert')
        pert = lyz.getperttime(pxi, mrkdata, thresholds.loc[thresholds.subject == subject_index + 1])
        print('Loading FPlate: ', file)
        fplate = vn.read_vicon_XYZ_file(vn.getpath(fppath, subject, file))
        fpt = np.arange(0, len(fplate.values[:, 0]) / 1000, 1 / 1000)  # fp time
        lftstept, rftstept = lyz.stepped(fplate.iloc[:, fplatecolumnind])
        # todo now label everything during stepping as 2

        # todo and then go through perturbations and label onset until end or step as 1

    for k in range(0, len(time_sets.values[:, 0])):
        # todo figure out when to end spliced marker data!
        # get the perturbation time onset that was delivered AFTER start of sit2stand
        if file[0:4] == 'pert':
            if not [x for x in pert.values[:, 0] - time_sets.values[k, 0] if x > 0]:
                print("perturbation onset before sit2stand")
                continue
            else:
                here = int(np.argmin([x for x in pert.values[:, 0] - time_sets.values[k, 0] if x > 0]) +
                           len([x for x in pert.values[:, 0] - time_sets.values[k, 0] if
                                x < 0]))  # index of pert time to use
            if pert.values[here, 0] > time_sets.values[k, 1]:
                print("perturbation onset was before sit2stand AND wrong selected pert cycle, continue")
                continue
            if pert.values[here, 2] == 2.0:
                # TODO label as 0: no perturbation
                cuttohere = time_sets.values[k, 1]
            # todo check if time_set.values[k,1] is b4 or after end of pert time choose later one!
            cuttohere = pert.values[here, 1]
        else:  # baseline data:
            here = k
            cuttohere = time_sets.values[k, 1]

        # center marker data based on feet position at start of sit2stand
        m = mrkdata.values[
            np.argmin(np.abs(time_sets.values[k, 0] - mt)): np.argmin(np.abs(cuttohere - mt)), :]
        m = lyz.recenterMrk(m)

    #TODO label
    # 0: no perturbation
    # 1: identify perturbation durations
    # 2: use gndForces to determine if 1(perturbedbutstable) or 2(falling)
    # filter EMG?