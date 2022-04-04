# Loads raw mrk, fplate, emg, pxi -> analyze data based on perturbation data -> export pickle
# user will only be asked to select marker data

# Affiliation : ROAR Lab
# Project: TPAD sit-to-stand: study01
# Author: Tatiana D. Luna
# Date: 1.24.22

# Import Libraries needed:
import matplotlib.pyplot as plt
import pandas
import tools.vicon as vn
import pickle as pkl
import tools.analyze as lyz
import numpy as np
import seaborn as sns

# File Paths from DESKTOP:
emgpath = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/EMG Export/'
fppath = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Force Plate Export/'
pxipath = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/PXI/'
subfile = 'D:/GitHub Content/RobUST/RobUST/TPAD-subjects.xlsx'
savehere = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Python Data Output/Output Variables'
dumphere = 'C:/Users/lunat/Google Drive/ROAR LAB/ATPAD/Study01-SittoStand/Python Data Output/pkl dumps'

# File Paths from LAPTOP:
# emgpath = 'G:/My Drive/ROAR LAB/ATPAD/Study01-SittoStand/EMG Export/'
# fppath = 'G:/My Drive/ROAR LAB/ATPAD/Study01-SittoStand/Force Plate Export/'
# pxipath = 'G:/My Drive/ROAR LAB/ATPAD/Study01-SittoStand/PXI/'
# subfile = 'C:/Users/User/Documents/GitHub/RobUST/TPAD-subjects.xlsx'

# Subject info:
sub_info = pandas.read_excel(subfile, sheet_name=0, header=0, usecols="B,D,E,F,G,H", nrows=16)
thresholds = pandas.read_excel(subfile, sheet_name='Thresholds')

filenames = vn.fxn_select_files()  # Ask user to select raw marker files to load
stringname = ''.join(filenames[0])
subject = stringname[stringname.index("sub"):stringname.index("sub") + 5]  # string
subject_index = int(subject[3:]) - 1  # subject_index = python index of subject, integer value
mrkpath = stringname[0:stringname.index("sub") + 6]
print(subject)

if subject_index == 0:
    fplatecolumnind = [2, 6, 7, 11, 15, 16]
else:
    fplatecolumnind = [2, 6, 7, 17, 21, 22]

muscles = ('lowrbck', 'abs', 'rectus_fem', 'bicep_fem', 'medial_gast', 'tbialis_ant')
fplatenames = ('LFz', 'Lcopx', 'Lcopy', 'RFz', 'Rcopx', 'Rcopy', 'LFznorm', 'RFznorm')

# open files selected and the associated EMG, Forceplate and PXI file *********************
for i in range(0, len(filenames)):
    stringname = ''.join(filenames[i])  # converts to string
    # find index of file name only (after sub until before .csv)
    file = stringname[stringname.index("sub") + 6:-4]
    print('Loading MRK: ', file)
    mrkdata = vn.read_vicon_XYZ_file(filenames[i])
    time_sets = lyz.splitsit2stand(mrkdata, subject_index)  # time (s) of when sub starts&finishes standing
    print('Loading FPlate: ', file)
    fplate = vn.read_vicon_XYZ_file(vn.getpath(fppath, subject, file))
    print('Loading EMG: ', file)
    emg = vn.read_emg_csv(vn.getpath(emgpath, subject, file))
    # time of mrk, fplate, emg data
    mt = np.linspace(0, len(mrkdata.values[:, 0]) / 200, len(mrkdata.values[:, 0]))  # mrk time
    et = np.linspace(0, len(emg.values[:, 0]) / 2000, len(emg.values[:, 0]))  # emg time
    fpt = np.linspace(0, len(fplate.values[:, 0]) / 1000, len(fplate.values[:, 0]))  # fp time

    if file[0:4] == 'pert':  # load perturbation pxi files:
        pxi = vn.read_pxi_txt(pxipath + subject + '/' + subject + '_' + file[0:4] + '_' + file[4:6] + '.txt')
        # perturbation onsets, endtime, perttype, pertvalue:
        pert = lyz.getperttime(pxi, mrkdata, thresholds.loc[thresholds.subject == subject_index + 1])
        its = 0
        for k in range(0, len(time_sets.values[:, 0])):
            # get the perturbation time onset that was delivered AFTER start of sit2stand
            if not [x for x in pert.values[:, 0] - time_sets.values[k, 0] if x > 0]:
                print("perturbation onset before sit2stand")
                continue
            else:
                here = int(np.argmin([x for x in pert.values[:, 0] - time_sets.values[k, 0] if x > 0]) +
                           len([x for x in pert.values[:, 0] - time_sets.values[k, 0] if
                                x < 0]))  # index of pert time to use
            if pert.values[here, 2] == 2.0:
                print("skipping visual perturbation")
                continue
            if pert.values[here, 0] > time_sets.values[k, 1]:
                print("perturbation onset was before sit2stand AND wrong selected pert cycle, continue")
                continue
           # center marker data based on feet position at start of sit2stand
            m = mrkdata.values[
                np.argmin(np.abs(time_sets.values[k, 0] - mt)): np.argmin(np.abs(pert.values[here, 1] - mt)), :]
            m = lyz.recenterMrk(m)
            # splice data and get output variables based on PERTURBATION times
            m = m[
                (np.argmin(np.abs(pert.values[here, 0] - mt)) -
                 np.argmin(np.abs(time_sets.values[k, 0] - mt))):, :]
            f = fplate.values[
                np.argmin(np.abs(pert.values[here, 0] - fpt)): np.argmin(np.abs(pert.values[here, 1] - fpt)),
                fplatecolumnind]
            # Fz normalized by total Fz:
            f = np.column_stack([f, f[:, [0, 3]] / np.column_stack([np.sum(f[:, [0, 3]], axis=1)])])
            em = emg.values[
                 np.argmin(np.abs(pert.values[here, 0] - et)): np.argmin(np.abs(pert.values[here, 1] - et)),
                 :] / maxEMG  # EMG data normalized by maximum of baseline
            angles, r_ang = lyz.getjointangles(
                pandas.DataFrame(data=m, columns=mrkdata.columns.tolist()), subject_index)
            outvars = lyz.analyze(angles, pandas.DataFrame(data=m, columns=mrkdata.columns.tolist()),
                                  pandas.DataFrame(data=f, columns=fplatenames),
                                  pandas.DataFrame(data=em, columns=emg.columns.tolist()),
                                  subject_index, pert.iloc[here, 2:])
            # resample data for plotting purposes later:
            xmn, ymn, zmn = lyz.getcofpelv(lyz.recenterMrk(mrkdata.values[
                                                           np.argmin(np.abs(time_sets.values[k, 0] - mt)):
                                                           np.argmin(np.abs(pert.values[here, 1] - mt)), :]),
                                           subject_index)
            com = lyz.resampvar(np.column_stack([xmn, ymn, zmn]), ('comx', 'comy', 'comz'), pert.iloc[here, :],
                                subject_index)
            r_emg = lyz.resampvar(emg.values[
                                  np.argmin(np.abs(time_sets.values[k, 0] - et)): np.argmin(
                                      np.abs(pert.values[here, 1] - et)),
                                  :] / maxEMG, muscles, pert.iloc[here, :], subject_index)
            frm = fplate.values[
                  np.argmin(np.abs(time_sets.values[k, 0] - fpt)): np.argmin(np.abs(pert.values[here, 1] - fpt)),
                  fplatecolumnind]
            frm = np.column_stack([frm, frm[:, [0, 3]] / np.column_stack([np.sum(frm[:, [0, 3]], axis=1)])])
            r_fp = lyz.resampvar(frm, fplatenames, pert.iloc[here, :], subject_index)

            if its == 0:
                saveoutputs = outvars
                cutmarkerdata = {k: m}
                cutfplatedata = {k: f}
                cutemgdata = {k: em}
                perturbationtimes = {k: pert}
                # save resampled data:
                resamp_com = {k: com}
                resamp_emg = {k: r_emg}
                resamp_fp = {k: r_fp}
                resamp_ang = {k: r_ang}
                its = its + 1
            else:
                saveoutputs = pandas.concat([saveoutputs, outvars])
                cutmarkerdata[k] = m
                cutfplatedata[k] = f
                cutemgdata[k] = em
                perturbationtimes[k] = pert
                # save resampled data:
                resamp_com[k] = com
                resamp_emg[k] = r_emg
                resamp_fp[k] = r_fp
                resamp_ang[k] = r_ang
                its = its+1


    else:  # Baseline data:
        pxi = 'NA'
        pert = pandas.DataFrame(data=[[-1, 0, -1, 0, 0, 0]],
                                columns=('pertType', 'pertValue', 'pertDirt',
                                         'pertampVal', 'pertlvl', 'heightwhenpert'))
        # loop through sit2stand cycles and average output vars:
        for k in range(0, len(time_sets.values[:, 0])):
            # splice data based on sit2stand start and end times
            m = mrkdata.values[
                np.argmin(np.abs(time_sets.values[k, 0] - mt)): np.argmin(np.abs(time_sets.values[k, 1] - mt)), :]
            m = lyz.recenterMrk(m)  # center marker data based on feet position at start of sit2stand
            f = fplate.values[
                np.argmin(np.abs(time_sets.values[k, 0] - fpt)): np.argmin(np.abs(time_sets.values[k, 1] - fpt)),
                fplatecolumnind]
            f = np.column_stack([f, f[:, [0, 3]] / np.column_stack([np.sum(f[:, [0, 3]], axis=1)])])
            em = emg.values[
                 np.argmin(np.abs(time_sets.values[k, 0] - et)): np.argmin(np.abs(time_sets.values[k, 1] - et)),
                 :]
            maxE = np.max(em, axis=0)
            angles, r_ang = lyz.getjointangles(
                pandas.DataFrame(data=m, columns=mrkdata.columns.tolist()), subject_index)
            outvars = lyz.analyze(angles, pandas.DataFrame(data=m, columns=mrkdata.columns.tolist()),
                                  pandas.DataFrame(data=f, columns=fplatenames),
                                  pandas.DataFrame(data=em, columns=emg.columns.tolist()),
                                  subject_index, pert)
            # resample for plotting purposes:
            r_fp = lyz.resampvar(f, fplatenames, pert, subject_index)
            r_emg = lyz.resampvar(em, muscles, pert, subject_index)
            xmn, ymn, zmn = lyz.getcofpelv(m, subject_index)
            com = lyz.resampvar(np.column_stack([xmn, ymn, zmn]), ('comx', 'comy', 'comz'), pert, subject_index)

            if k == 0:
                maxEMG = maxE
                saveoutputs = outvars.values
                cutmarkerdata = {k: m}
                cutfplatedata = {k: f}
                cutemgdata = {k: em}
                perturbationtimes = {k: pert}
                # save resampled data:
                resamp_com = {k: com}
                resamp_emg = {k: r_emg}
                resamp_fp = {k: r_fp}
                resamp_ang = {k: r_ang}
            else:
                maxEMG = np.mean([maxEMG, maxE], axis=0)
                saveoutputs = np.mean([saveoutputs, outvars.values], axis=0)
                cutmarkerdata[k] = m
                cutfplatedata[k] = f
                cutemgdata[k] = em
                perturbationtimes[k] = pert
                # save resampled data:
                resamp_com[k] = com
                resamp_emg[k] = r_emg
                resamp_fp[k] = r_fp
                resamp_ang[k] = r_ang

        # once baseline sit2stand loops are over:
        saveoutputs = pandas.DataFrame(data=saveoutputs, columns=outvars.columns.values.tolist())

    # save file info before loading next file
    if i == 0:
        #continue
        saveoutputs.to_csv((savehere + '/' + subject + '.csv'))
    else:
        saveoutputs.to_csv((savehere + '/' + subject + '.csv'), mode='a', header=False)

    compiledcutData = [sub_info.iloc[subject_index, :], thresholds.loc[thresholds.subject == subject_index + 1],
                       cutmarkerdata, cutfplatedata, cutemgdata, saveoutputs]  # store as list
    compiledresampData = [sub_info.iloc[subject_index, :], thresholds.loc[thresholds.subject == subject_index + 1],
                          resamp_com, resamp_fp, resamp_emg, resamp_ang]  # store as list
    outfile = open((dumphere + '/' + subject + '_' + file), 'wb')
    pkl.dump(compiledcutData, outfile)
    outfile.close()
    outfile = open((dumphere + '/' + subject + '_resamp_' + file), 'wb')
    pkl.dump(compiledresampData, outfile)
    outfile.close()
    print('pickle done for:', file)
    k = 0
    # end of file for loop

print('Done with all files')
