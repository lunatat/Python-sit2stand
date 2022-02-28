import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import signal
import matplotlib.pyplot as plt
import pandas
from scipy.spatial.transform import Rotation as R
import seaborn as sns
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.geometry import shape

# Fxns to analyze sit to stand data

def recenterMrk(data):
    """
    average the feet markers x y z and recenters all marker data to be about center of feet position

    :param data: marker array
    :return: centered marker at base of support - center of feet
    """
    out = data
    # out = data.values  # uncomment if data input is dataframe
    # out = np.vstack(out[:, :]).astype(np.float)  # converts to float array
    mnx = np.mean(out[0, 57:-1:3])  # slicing syntax list[start:stop:step]
    mny = np.mean(out[0, 58:-1:3])
    mnz = np.mean(out[0, 59::3])
    out[:, 0::3] = out[:, 0:-1:3] - mnx
    out[:, 1::3] = out[:, 1:-1:3] - mny
    out[:, 2::3] = out[:, 2::3] - mnz
    centered = out
    return centered


##########################################################################

def splitsit2stand(data, subject_index) -> pandas.DataFrame:
    """
    get pelvis center from marker data to split sit2stand and get the index pts
    Indexing based on vicon skeleton 'fullbody_3DFF_editting'

    :param data: marker data- use center of pelvis to split
    :param subject_index: python index value of subject- integer value
    :return: time_set: pandas dataframe with time pts where sit-2-stand began and ended
    """
    out = data.values
    out = np.vstack(out[:, :]).astype(np.float)  # converts to float array
    xmn, ymn, zmn = getcofpelv(out, subject_index)
    negzmn = -1 * zmn
    h = np.min(negzmn) + 0.5 * (np.max(negzmn) - np.min(negzmn))  # 50% of the max-min height
    sitting, properties = signal.find_peaks(negzmn, height=h, distance=500)  # finding valleys aka when seated
    standing, properties = signal.find_peaks(zmn, height=-h, distance=400)  # finding peaks aka when standing
    # find closest point to the left of standing - use that as the start of the sitting index
    seated = np.empty(shape=[len(standing), 1], dtype=int)
    for k in range(0, len(standing)):
        seated[k] = sitting[np.argmin(standing[k] - sitting[(standing[k] - sitting) > 0])]
    seated = np.unique(seated)  # removes duplicates

    # # plot 'peaks'
    plt.plot(zmn)
    plt.plot(standing, zmn[standing], "x")
    plt.plot(seated, zmn[seated], "x")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # find velocity: slice data based on vel thresholds
    dt = 1 / 200  # vicon markers sampled at 200Hz
    velz = (zmn[1:-1] - zmn[0:-2]) / dt
    vonset = 20  # onset threshold mm/frame
    slidingwind = 20
    stance = np.empty(shape=[len(seated), 1], dtype=int)
    onset = np.empty(shape=[len(seated), 1], dtype=int)
    near_stance = np.empty(shape=[len(seated), 1], dtype=int)

    for k in range(0, len(seated)):
        if k == len(seated) - 1:
            stance[k] = seated[k] + np.argmax(zmn[seated[k]:-1])
        else:
            stance[k] = seated[k] + np.argmax(zmn[seated[k]:seated[k + 1]])  # index of standing
        maxpt = np.argmax(velz[seated[k]:stance.item(k)]) + seated[
            k]  # index at max v - should be in the beginning of standing up
        # sliding window from maxpt backwards - stop if we reach sitting[k]: searching for vonset
        while maxpt >= seated[k]:
            if np.min(np.absolute(velz[(maxpt - slidingwind):maxpt] - vonset)) <= 2:
                # onset sit2stand
                onset[k] = maxpt - slidingwind + np.argmin(np.absolute(velz[(maxpt - slidingwind):maxpt] - vonset))
                break
            else:
                maxpt = maxpt - slidingwind
                if maxpt < seated[k]:
                    onset[k] = seated[k]
                    break
        while maxpt <= stance[k]:
            if np.min(np.absolute(velz[maxpt:(maxpt + slidingwind)] - vonset)) <= 2:
                # onset sit2stand
                near_stance[k] = maxpt + np.argmin(np.absolute(velz[maxpt:(maxpt + slidingwind)] - vonset))
                break
            else:
                maxpt = maxpt + slidingwind
                if maxpt > stance[k]:
                    near_stance[k] = stance[k]
                    break

    # plot segmented data:
    plt.plot(zmn)
    plt.plot(seated, zmn[seated], "x")
    plt.plot(onset, zmn[onset], "o")
    plt.plot(near_stance, zmn[near_stance], "o")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    # # plot velocity:
    # plt.plot(velz)
    # plt.plot(seated, velz[seated], "x")
    # plt.plot(onset, velz[onset], "o")
    # plt.plot(stance, velz[stance], "o")
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()
    # find true stopping point? when v is slows down to vthreshold?

    # onset values are in terms of marker index change to time in order to sync with other values
    time_set = pandas.DataFrame(data=np.column_stack((onset, near_stance)) * dt, columns=('starttime', 'endtime'))
    return time_set


##########################################################################

def getoutputvars(var):
    """
    calculates variables' mean, std, max, min, range, variance, total excursion

    :param data:
    :return: outputs: dataframe of 1 x (size of vars * 7)
    """
    vr = var.values
    vr = np.vstack(vr[:, :]).astype(np.float)
    out = np.empty(shape=[7, len(vr[0, :])])
    out = np.stack([np.mean(vr, axis=0), np.std(vr, axis=0), np.max(vr, axis=0), np.min(vr, axis=0),
                    np.abs(np.max(vr, axis=0) - np.min(vr, axis=0)), np.var(vr, axis=0),
                    np.sum(np.abs(vr[1:-1, :] - vr[0:-2, :]), axis=0)])
    names = ('_avg', '_std', '_max', '_min', '_rnge', '_var', '_totex')
    var_names = (v + n for n in names for v in var.columns.values.tolist())
    outputs = pandas.DataFrame(data=[out.flatten()], columns=var_names)
    return outputs


##########################################################################

def getperttime(data, mrk, threshold):
    """
    pertval - has forces in terms of %
    pertlvl: 0 = baseline, 1=first, 2=thres, 3=bynd, 4=NA

    :param data: pxi dataframe
    :param mrk: is marker dataframe xyz position data
    :param threshold: threshold dataframe of subject's perturbation threshold info
    :return: time_set: perturbation onset- based on tension/feetposition - and endofperttime
    """
    # 1 check which type of perturbation  perttype(0=pelv, 1=treadmill, 2=visual)
    # 2 get onset/endttme of perturbations: pelv-tensiondata, visual-perton, treadmill-feetYvelocity
    # 3 calculate current onset position of pelvis/ maximum pelvic height
    # 4  tension data checks valleys- of averaged current tension - finds end point by checking desired T
    #       treadmill onset is determined by velocity y of LEFT heel!

    out = mrk.values
    out = np.vstack(out[:, :]).astype(np.float)  # converts to float array
    xmn, ymn, zmn = getcofpelv(out, threshold.values[0, 0] - 1)
    # fq of pxi is 200hz double check
    pd = 100  # pertduration = 0.5s *200 frame/s
    dt = 1 / 200
    out = data.values
    out = np.vstack(out[:, :]).astype(np.float)  # converts to float array
    perton = np.asarray(np.nonzero(np.diff(data["pertOn"]) == 1)) + 1  # index of pert is on
    # initialze variables
    onset = np.empty(shape=[len(perton[0]), 1], dtype=int)
    endp = np.empty(shape=[len(perton[0]), 1], dtype=int)
    perttype = np.empty(shape=[len(perton[0]), 1], dtype=float)
    pertvalue = np.empty(shape=[len(perton[0]), 1], dtype=float)
    pertdirt = np.empty(shape=[len(perton[0]), 1], dtype=float)
    pertval = np.empty(shape=[len(perton[0]), 1], dtype=float)
    pertlvl = np.empty(shape=[len(perton[0]), 1], dtype=float)  # 0 = baseline, 1=first, 2=thres, 3=bynd, 4=NA
    heightwhenpert = np.empty(shape=[len(perton[0]), 1], dtype=float)
    w = threshold.values[0, 6] * 100 / (9.81 * threshold.values[0, 2])  # weight of subject:
    j = 0
    dest = data.columns.get_loc("pel_des_T1")
    ct = data.columns.get_loc("pel_act_T1")
    f = data.columns.get_loc("PertampF")
    for k in perton[0]:
        if data["perttype"][k] == 0:  # pelvis pert
            # check tension data for true onset
            if data["pertDirection"][k] == 0.0:  # south
                cables = [x + ct for x in [2, 3, 6, 7]]  # python is dumb this is att + []
                # check if force is past subject's threshold!
                if data.values[k, f] <= threshold.values[0, -1]:
                    if round(data.values[k, f]*100/(w*9.81)) == 5 or round(data.values[k, f]*100/(w*9.81)) == 6:
                        pertlvl[j] = 1
                    elif round(data.values[k, f]*100/(w*9.81)) == threshold.values[0, 2]:
                        pertlvl[j] = 2
                    else:
                        pertlvl[j] = 4
                else:
                    pertlvl[j] = 3
            elif data["pertDirection"][k] == 4.0:  # north
                cables = [x + ct for x in [0, 1, 4, 5]]
                if data.values[k, f] <= threshold.values[0, -2]:
                    if round(data.values[k, f]*100/(w*9.81)) == 5:
                        pertlvl[j] = 1
                    elif round(data.values[k, f]*100/(w*9.81)) == threshold.values[0, 1]:
                        pertlvl[j] = 2
                    else:
                        pertlvl[j] = 4
                else:
                    pertlvl[j] = 3
            meanT = np.mean(out[k:k + 200, cables], 1)
            valleys, prop = signal.find_peaks(-meanT, distance=60)
            mxpkdist = np.argmax(valleys[1:] - valleys[0:-1])
            onset[j] = k + valleys[mxpkdist]
            meandesT = np.mean(out[k:k + 200, dest:dest + 8], 1)
            # find when meandestT is = to 15, then search forward for closest valley to it-thats endpoint:
            endp[j] = valleys[np.argmin(np.argmin(meandesT - 15) - valleys)] + k
            # onset[j] = k + 40 + np.argmin(meanT[40:pd])
            # NOT FOOL PROOF check plots
            # endp[j] = onset.item(j) + 100 + np.argmin(np.abs(meanT(onset.item(j))-meanT[100:]))
            plt.plot(out[onset.item(j):endp.item(j), dest:dest + 8])
            plt.plot(out[onset.item(j):endp.item(j), ct:ct + 8])
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            pertvalue[j] = data["PertampF"][onset.item(j)]
            pertval[j] = np.round((pertvalue[j] * 100) / (9.81 * w))
        elif data["perttype"][k] == 1:  # treadmill
            # check Left heel velocity for onset
            # delay is about 100 frames from onset
            vely = (mrk.values[k + 1:k + 300, -2] - mrk.values[k:k + 299, -2]) / dt
            if data["pertDirection"][k] == 4.0:  # north
                onset[j] = k + 100 + np.argmin(vely[100:150])  # NOT FOOL PROOF check plots
                endp[j] = onset.item(j) + 150 + np.argmin(vely[150:])
                # plt.plot(vely)
                # plt.plot(np.argmin(vely[100:150]) + 100, vely[100 + np.argmin(vely[100:150])], "o")
                # plt.plot(np.argmin(vely[150:]) + 150, vely[150 + np.argmin(vely[150:])], "o")
                if data.values[k, f + 1] <= threshold.values[0, 3]:
                    if data.values[k, f+1] == 0.4:
                        pertlvl[j] = 1
                    elif data.values[k, f+1] == threshold.values[0, 3]:
                        pertlvl[j] = 2
                    else:
                        pertlvl[j] = 4
                else:
                    pertlvl[j] = 3
            elif data["pertDirection"][k] == 0.0:  # south
                onset[j] = k + 100 + np.argmax(vely[100:150])
                endp[j] = onset.item(j) + 150 + np.argmax(vely[150:])
                if data.values[k, f + 1] <= threshold.values[0, 4]:
                    if data.values[k, f+1] == 0.4:
                        pertlvl[j] = 1
                    elif data.values[k, f+1] == threshold.values[0, 4]:
                        pertlvl[j] = 2
                    else:
                        pertlvl[j] = 4
                else:
                    pertlvl[j] = 3
            plt.plot(mrk.values[onset.item(j):endp.item(j), -2])  # left foot heel y
            plt.plot(mrk.values[onset.item(j):endp.item(j)+100, 64])  # right foot heel y
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            pertvalue[j] = data["PertAccel"][onset.item(j)]
            pertval[j] = pertvalue[j]
        else:  # visual disturbances:
            onset[j] = k
            endp[j] = k + 150
            pertvalue[j] = data["PertVisual"][onset.item(j)]
            pertval[j] = pertvalue[j]
            pertlvl[j] = 4
        heightwhenpert[j] = (zmn[onset.item(j)] - np.min(zmn)) / (np.max(zmn) - np.min(zmn))
        perttype[j] = data["perttype"][k]
        pertdirt[j] = data["pertDirection"][k]
        j = j + 1

    ts = np.column_stack((onset, endp)) * dt  # time data
    # get pelvis position data
    time_set = pandas.DataFrame(data=np.column_stack((ts, perttype, pertval, pertdirt, pertvalue, pertlvl, heightwhenpert)),
                                columns=('pertStart', 'pertEnd', 'pertType', 'pertValue', 'pertDirt',
                                         'pertampVal', 'pertlvl', 'heightwhenpert'))
    return time_set


##########################################################################
def analyze(angles, mrk, fplate, emg, subject_index, pertinfo):
    """

    :param mrk:
    :param fplate:
    :param emg:
    :return:
    """

    var_ang = getoutputvars(angles)
    var_fplate = getoutputvars(fplate)
    var_si = getSI(fplate)  # mean symmetry index of gnd rxn forces
    var_emg = getemgoutputs(emg)
    var_mos = getmos(mrk, subject_index)
    # compile into 1xn dataframe of subinfo, pertinfo, all output variables
    subinfo = pandas.DataFrame(data=[subject_index + 1], columns=('subject',))
    if isinstance(pertinfo, pd.Series):
        pertinfo = pertinfo.to_frame().transpose()
    output = pandas.concat([subinfo, var_si, var_mos, var_emg, var_ang, var_fplate], axis=1)
    output[pertinfo.columns.to_list()] = pertinfo.values
    return output

##########################################################################
def getjointangles(mrk, subject_index):
    """ calculates ankle, knee, hip, trunk joint angles
    creates segment axis frame-> rotation matrix = axisframe/segmentnexttoaxis
    The only exception is the trunk joint angles are relative to the first trunk position

    :param mrk: dataframe
    :param subject_index: python subject index
    :return: joint angles
    """
    # splice marker data:
    out = mrk.values
    out = np.vstack(out[:, :]).astype(np.float)  # converts to float array
    rftc = np.column_stack(
        [np.mean(out[:, 57:64:3], axis=1), np.mean(out[:, 58:65:3], axis=1), np.mean(out[:, 59:66:3], axis=1)])
    lftc = np.column_stack(
        [np.mean(out[:, 66:73:3], axis=1), np.mean(out[:, 67:74:3], axis=1), np.mean(out[:, 68::3], axis=1)])
    rtoe = out[:, 57:60] - out[:, 63:66]  # vector heel2toe = toe - heel
    rftout = out[:, 60:63] - rftc  # vector from center of ft to foot out = ftout - center
    ltoe = out[:, 66:69] - out[:, 72:]  # vector heel2toe = toe - heel
    lftout = out[:, 69:72] - lftc
    ranktoknee = out[:, 36:39] - out[:, 42:45]  # kneeout - ankle
    lanktoknee = out[:, 48:51] - out[:, 54:57]
    rkneein2out = out[:, 36:39] - out[:, 39:42]  # rkneeout - rkneein
    lkneeout2in = out[:, 51:54] - out[:, 48:51]  # lkneein - lkneeout
    rknee2hip = out[:, 33:36] - out[:, 36:39]  # rleghip - rkneeout
    lknee2hip = out[:, 45:48] - out[:, 48:51]  # lleghip - lkneeout
    # subjects that used pelvic rods:
    usedrods = list(range(0, 6))
    usedrods.append(7)
    if subject_index in usedrods:
        # average the 4 true pelv markers for each row
        xmn = np.mean(out[:, [15, 18, 27, 30]], axis=1)
        ymn = np.mean(out[:, [16, 19, 28, 31]], axis=1)
        zmn = np.mean(out[:, [17, 20, 29, 32]], axis=1)
        pelvantl = out[:, 30:33]
        pelvantr = out[:, 27:30]
        frontpelv = np.column_stack(
            [np.mean(out[:, [27, 30]], axis=1), np.mean(out[:, [28, 31]], axis=1), np.mean(out[:, [29, 32]], axis=1)])
    else:
        # average all pelvis markers for each row
        xmn = np.mean(out[:, 15:31:3], axis=1)
        ymn = np.mean(out[:, 16:32:3], axis=1)
        zmn = np.mean(out[:, 17:33:3], axis=1)
        pelvantl = out[:, 24:27]
        pelvantr = out[:, 21:24]
        frontpelv = np.column_stack(
            [np.mean(out[:, [21, 24]], axis=1), np.mean(out[:, [22, 25]], axis=1), np.mean(out[:, [23, 26]], axis=1)])
    midpelv = np.column_stack([xmn, ymn, zmn])
    frontpelv = frontpelv - midpelv
    pelvantl = pelvantl - midpelv
    pelvantr = pelvantr - midpelv
    midtrunk = np.column_stack(
        [np.mean(out[:, 3:13:3], axis=1), np.mean(out[:, 4:14:3], axis=1), np.mean(out[:, 5:15:3], axis=1)])
    midtrunk = midtrunk - midpelv
    l2rshld = out[:, 9:12] - out[:, 12:15]  # vector from left2rshoulder
    # initialize joint angle variables
    ang_r_ank = np.empty(shape=rtoe.shape, dtype=float)
    ang_l_ank = np.empty(shape=rtoe.shape, dtype=float)
    ang_r_knee = np.empty(shape=rtoe.shape, dtype=float)
    ang_l_knee = np.empty(shape=rtoe.shape, dtype=float)
    ang_r_hip = np.empty(shape=rtoe.shape, dtype=float)
    ang_l_hip = np.empty(shape=rtoe.shape, dtype=float)
    ang_trunk = np.empty(shape=rtoe.shape, dtype=float)
    for k in range(0, len(out[:, 1])):
        # CALCULATE ANKLE JOINT ANGLE ***********
        # y = vector from heal to toe
        yftr = rtoe[k, :] / np.sqrt(np.sum(np.power(rtoe[k, :], 2)))
        yftl = ltoe[k, :] / np.sqrt(np.sum(np.power(ltoe[k, :], 2)))
        # zftR = ftout x y
        zftr = np.cross(rftout[k, :] / np.sqrt(np.sum(np.power(rftout[k, :], 2))), yftr)
        zftr = zftr / np.sqrt(np.sum(np.power(zftr, 2)))
        # zftl = y x ftout
        zftl = np.cross(yftl, lftout[k, :] / np.sqrt(np.sum(np.power(lftout[k, :], 2))))
        zftl = zftl / np.sqrt(np.sum(np.power(zftl, 2)))
        # xft = y cross z
        xftr = np.cross(yftr, zftr) / np.sqrt(np.sum(np.power(np.cross(yftr, zftr), 2)))
        xftl = np.cross(yftl, zftl) / np.sqrt(np.sum(np.power(np.cross(yftl, zftl), 2)))
        # Feet axis frames:
        raxisft = [xftr, yftr, zftr]
        laxisft = [xftl, yftl, zftl]
        # calculate calf axis frames:
        # zknee = vector from ankle to kneeout
        zkneer = ranktoknee[k, :] / np.sqrt(np.sum(np.power(ranktoknee[k, :], 2)))
        zkneel = lanktoknee[k, :] / np.sqrt(np.sum(np.power(lanktoknee[k, :], 2)))
        # y = z x kneevector
        ykneer = np.cross(zkneer, rkneein2out[k, :] / np.sqrt(np.sum(np.power(rkneein2out[k, :], 2))))
        ykneer = ykneer / np.sqrt(np.sum(np.power(ykneer, 2)))
        ykneel = np.cross(zkneel, lkneeout2in[k, :] / np.sqrt(np.sum(np.power(lkneeout2in[k, :], 2))))
        ykneel = ykneel / np.sqrt(np.sum(np.power(ykneel, 2)))
        # x = y x z
        xkneer = np.cross(ykneer, zkneer) / np.sqrt(np.sum(np.power(np.cross(ykneer, zkneer), 2)))
        xkneel = np.cross(ykneel, zkneel) / np.sqrt(np.sum(np.power(np.cross(ykneel, zkneel), 2)))
        # calf axis frames:
        raxisclf = [xkneer, ykneer, zkneer]
        laxisclf = [xkneel, ykneel, zkneel]
        # ankle angle = from rotation matrix R = calf/foot matrices
        rmatank_r = R.from_matrix(np.matrix(raxisclf) @ inv(np.matrix(raxisft)))
        rmatank_l = R.from_matrix(np.matrix(laxisclf) @ inv(np.matrix(laxisft)))
        ang_r_ank[k, :] = rmatank_r.as_euler('xyz', degrees=True)
        ang_l_ank[k, :] = rmatank_l.as_euler('xyz', degrees=True)
        # CALCULATE KNEE Joint ANGLE ************
        zhipr = rknee2hip[k, :] / np.sqrt(np.sum(np.power(rknee2hip[k, :], 2)))
        zhipl = lknee2hip[k, :] / np.sqrt(np.sum(np.power(lknee2hip[k, :], 2)))
        # y = z x kneevector
        yhipr = np.cross(zhipr, rkneein2out[k, :] / np.sqrt(np.sum(np.power(rkneein2out[k, :], 2))))
        yhipr = yhipr / np.sqrt(np.sum(np.power(yhipr, 2)))
        yhipl = np.cross(zhipl, lkneeout2in[k, :] / np.sqrt(np.sum(np.power(lkneeout2in[k, :], 2))))
        yhipl = yhipl / np.sqrt(np.sum(np.power(yhipl, 2)))
        # x = y x z
        xhipr = np.cross(yhipr, zhipr) / np.sqrt(np.sum(np.power(np.cross(yhipr, zhipr), 2)))
        xhipl = np.cross(yhipl, zhipl) / np.sqrt(np.sum(np.power(np.cross(yhipl, zhipl), 2)))
        # hip axis frames:
        raxiship = [xhipr, yhipr, zhipr]
        laxiship = [xhipl, yhipl, zhipl]
        # knee angle- from rotation matrices R = hip/calf matrices
        rmatknee_r = R.from_matrix(np.matrix(raxiship) @ inv(np.matrix(raxisclf)))
        rmatknee_l = R.from_matrix(np.matrix(laxiship) @ inv(np.matrix(laxisclf)))
        ang_r_knee[k, :] = rmatknee_r.as_euler('xyz', degrees=True)
        ang_l_knee[k, :] = rmatknee_l.as_euler('xyz', degrees=True)
        # CALCULATE HIP JOIN ANGLE ******************
        # z = pelvantr x pelvantl
        zpelv = np.cross(pelvantr[k, :], pelvantl[k, :]) / np.sqrt(
            np.sum(np.power(np.cross(pelvantr[k, :], pelvantl[k, :]), 2)))
        ypelv = frontpelv[k, :] / np.sqrt(np.sum(np.power(frontpelv[k, :], 2)))
        # x = y x z
        xpelv = np.cross(ypelv, zpelv) / np.sqrt(np.sum(np.power(np.cross(ypelv, zpelv), 2)))
        axispelv = [xpelv, ypelv, zpelv]
        rmathip_r = R.from_matrix(np.matrix(axispelv) @ inv(np.matrix(raxiship)))
        rmathip_l = R.from_matrix(np.matrix(axispelv) @ inv(np.matrix(laxiship)))
        ang_r_hip[k, :] = rmathip_r.as_euler('xyz', degrees=True)
        ang_l_hip[k, :] = rmathip_l.as_euler('xyz', degrees=True)
        # CALCULATE TRUNK ANGLE *****************
        ztrunk = midtrunk[k, :] / np.sqrt(np.sum(np.power(midtrunk[k, :], 2)))
        ytrunk = np.cross(ztrunk, l2rshld[k, :] / np.sqrt(np.sum(np.power(l2rshld[k, :], 2))))
        ytrunk = ytrunk / np.sqrt(np.sum(np.power(ytrunk, 2)))
        xtrunk = np.cross(ytrunk, ztrunk) / np.sqrt(np.sum(np.power(np.cross(ytrunk, ztrunk), 2)))
        if k == 0:
            basetrunk = [xtrunk, ytrunk, ztrunk]
            rmattrk = R.from_matrix(np.matrix(basetrunk) @ inv(np.matrix(basetrunk)))
        else:
            axistrunk = [xtrunk, ytrunk, ztrunk]
            rmattrk = R.from_matrix(np.matrix(axistrunk) @ inv(np.matrix(basetrunk)))
        ang_trunk[k, :] = rmattrk.as_euler('xyz', degrees=True)
    # compile angles into dataframe
    ang = np.column_stack([ang_r_ank, ang_l_ank, ang_r_knee, ang_l_knee, ang_r_hip, ang_l_hip, ang_trunk])
    ang_names = ('r_ankle', 'l_ankle', 'r_knee', 'l_knee', 'r_hip', 'l_hip', 'trunk')
    angles = pandas.DataFrame(data=ang, columns=(a + xyz for a in ang_names for xyz in ('_x', '_y', '_z')))
    # plot angles to verify
    g = sns.relplot(data=pandas.DataFrame(data=ang[:, 0:-1:3], columns=ang_names))
    g.set_xlabels("Cycle")
    g.set_ylabels("Joint Angles (degrees)")
    g.map(sns.kdeplot)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    # resample data for plotting
    cycle = np.arange(1, 101, 1)
    d = signal.resample_poly(ang[:, 0:-1:3], 100, len(ang[:, 0]), axis=0, padtype='line')
    resamp_ang = pandas.DataFrame(data=np.column_stack([cycle, d]),
                                  columns=(('Cycle',) + ang_names))
    resamp_ang = pandas.melt(resamp_ang, id_vars=('Cycle'), var_name='Joint', value_name='Angle_x')
    resamp_ang_y = pandas.DataFrame(data=np.column_stack(
        [cycle, signal.resample_poly(ang[:, 1:-1:3], 100, len(ang[:, 0]), axis=0, padtype='line')]),
        columns=(('Cycle',) + ang_names))
    resamp_ang_y = pandas.melt(resamp_ang_y, id_vars=('Cycle'), var_name='Joint', value_name='Angle_y')
    resamp_ang = resamp_ang.join(resamp_ang_y.Angle_y)
    resamp_ang_z = pandas.DataFrame(
        data=np.column_stack([cycle, signal.resample_poly(ang[:, 2::3], 100, len(ang[:, 0]), axis=0, padtype='line')]),
        columns=(('Cycle',) + ang_names))
    resamp_ang_z = pandas.melt(resamp_ang_z, id_vars=('Cycle'), var_name='Joint', value_name='Angle_z')
    resamp_ang = resamp_ang.join(resamp_ang_z.Angle_z)
    # the flex/ext angles are just the x angles but abs() and ankle angle is 90-angle
    resamp_ang_plot = pandas.DataFrame(data=np.column_stack([cycle, signal.resample_poly(
        np.column_stack([90 - ang[:, 0:4:3], np.abs(ang[:, 6:16:3]), ang[:, 18]]),
        100, len(ang[:, 0]), axis=0, padtype='line')]), columns=(('Cycle',) + ang_names))
    resamp_ang_plot = pandas.melt(resamp_ang_plot, id_vars=('Cycle'), var_name='Joint', value_name='Flex_Ext_Angle')
    resamp_ang = resamp_ang.join(resamp_ang_plot.Flex_Ext_Angle)
    resamp_ang['subject'] = subject_index + 1
    # sns.relplot(data=resamp_ang, x="Cycle", y="Flex_Ext_Angle", hue="Joint")
    return angles, resamp_ang


##########################################################################

def getemgoutputs(emg):
    """ calculate trapezoid integral, maximum, mean and std

    :param emg: filtered emg data
    :return: pandas dataframe of output emg muscles: mean std max and iemg
    """
    out = np.stack([np.mean(emg.values, axis=0), np.std(emg.values, axis=0),
                    np.max(emg.values, axis=0), np.trapz(emg.values, axis=0)])
    muscles = ('lowrbck', 'abs', 'rectus_fem', 'bicep_fem', 'medial_gast', 'tbialis_ant')
    names = ('_avg', '_std', '_max', '_iEMG')
    var_names = (musc + n for n in names for musc in muscles)
    outputs = pandas.DataFrame(data=[out.flatten()], columns=var_names)
    return outputs


##########################################################################

def getSI(fplate):
    """ symmetry index for gnd force z direction

    :param fplate:
    :return: pandas data frame of mean symmetry index
    """
    fz = np.absolute(fplate.values[:, [0, 3]])
    si = ((fz[:, 0] - fz[:, 1]) / np.mean(fz, axis=1)) * 100  # SI avg = (xleft-xright)/avg(xleft,xright) *100
    si = np.mean(si)
    si_df = pandas.DataFrame(data=[si], columns=('SI',))
    return si_df


##########################################################################
def getmos(mrk, subject_index):
    """
    calculating com based on center of pelvis
    Indexing based on vicon skeleton 'fullbody_3DFF_editting'

    :param mrk: dataframe of markerset using
    :return: pandas data frame of margin of stability in x,y and overall dist directions
    """

    # splice marker data:
    out = mrk.values
    out = np.vstack(out[:, :]).astype(np.float)  # converts to float array
    xmn, ymn, zmn = getcofpelv(out, subject_index)
    dt = 1 / 200  # vicon markers sampled at 200Hz
    velx = (xmn[1:-1] - xmn[0:-2]) / dt
    vely = (ymn[1:-1] - ymn[0:-2]) / dt
    g = 9.81 * 1000  # convert to mm
    l = 1.2 * zmn[0:-2]
    xcom = xmn[0:-2] + velx / np.sqrt(g / l)
    ycom = ymn[0:-2] + vely / np.sqrt(g / l)
    xcom_est = np.column_stack([xcom, ycom])  # xcom estimate = based on horizontal pelvis center
    # bosx = [rtoe(:, 1), rm(:, 1), rheal(:, 1), lheal(:, 1), lm(:, 1), ltoe(:, 1)];
    # bosy = [rtoe(:, 2), rm(:, 2), rheal(:, 2), lheal(:, 2), lm(:, 2), ltoe(:, 2)];
    bosx = out[:, [57, 60, 63, 72, 69, 66]]
    bosy = out[:, [58, 61, 64, 73, 70, 67]]
    mos = np.empty(shape=[len(xcom), 3], dtype=int)
    for k in range(0, len(out[:-2, 1])):
        bos = Polygon(np.column_stack([bosx[k, :], bosy[k, :]]))
        pt = Point(xcom_est[k, :])
        p1, p2 = nearest_points(bos.exterior, pt)
        mos[k, [0, 1]] = np.array([p1.x, p1.y]) - xcom_est[k, :]
        mos[k, 2] = np.sqrt((mos[k, 0]*mos[k, 0]) + (mos[k, 1]*mos[k, 1]))
        if pt.within(bos) == False:
            mos[k, 2] = -1*mos[k, 2]
    plt.plot(mos)
    plt.plot(xcom, ycom)
    plt.plot(bosx[-1], bosy[-1])
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    mos_df = pandas.DataFrame(data=[np.mean(mos, axis=0)], columns=('mosx', 'mosy', 'mos'))
    return mos_df


##########################################################################
def getcofpelv(out, subject_index):
    """calculates the center of pelvis x,y,z position
    Indexing based on vicon skeleton 'fullbody_3DFF_editting'

    :param subject_index: python index of subject #
    :param out:marker np. array
    :return: x,y,z mean of pelvis
    """
    usedrods = list(range(0, 6))
    usedrods.append(7)

    if subject_index in usedrods:
        # average 4 pelv markers for each row (don't include rods)
        xmn = np.mean(out[:, [15, 18, 27, 30]], axis=1)
        ymn = np.mean(out[:, [16, 19, 28, 31]], axis=1)
        zmn = np.mean(out[:, [17, 20, 29, 32]], axis=1)
    else:
        # average all pelvis markers for each row
        xmn = np.mean(out[:, 15:31:3], axis=1)
        ymn = np.mean(out[:, 16:32:3], axis=1)
        zmn = np.mean(out[:, 17:33:3], axis=1)

    return xmn, ymn, zmn


##########################################################################
def resampvar(var, var_names, pertinfo, subject_index):
    """

    :param pertinfo:
    :param var_names:
    :param var: to resample
    :return: resamp_var: resampled from 1 to 100 points in dataframe
    """
    # resample data for plotting
    cycle = np.arange(1, 101, 1)
    d = signal.resample_poly(var, 100, len(var[:, 0]), axis=0, padtype='line')
    resamp_var = pandas.DataFrame(data=np.column_stack([cycle, d]),
                                  columns=(('Cycle',) + var_names))
    resamp_var['subject'] = subject_index + 1
    if isinstance(pertinfo, pd.DataFrame):
        for k in range(0, len(pertinfo.columns.to_list())):
            resamp_var[pertinfo.columns.to_list()[k]] = pertinfo.values[0, k]
    else:
        for k in range(0, len(pertinfo.index.to_list())):
            resamp_var[pertinfo.index.to_list()[k]] = pertinfo.values[k]
    return resamp_var

##########################################################################
def stepped(fzdata):
    """
    :param fzdata: gndFz left and right
    :return: time point when indv. is unstable-steps
    """
    fpt = np.linspace(0, len(fzdata.values[:, 0]) / 1000, len(fzdata.values[:, 0]))  # fp time
    out = fzdata.values
    np.min(out[:, 0])
    zeroLfz = np.array(np.where(out[:, 0] >= -11))
    zeroRfz = np.array(np.where(out[:, 1] >= -11))
    lftstept = fpt[zeroLfz[0, :]]
    rftstept = fpt[zeroRfz[0, :]]
    return lftstept, rftstept

##########################################################################
def label_step(data, lftstept, rftstept, t):
    """ relabels data.stable to 2 if there was any lftstep or rftstep

    :param t: time of data
    :param rftstept: the time rft steps
    :param data: emg or mrk data
    :param lftstept: time lft steps
    :return: data with labeled : 2
    """
    if len(lftstept) != 0:
        # label here left foot step times
        indexlstep = np.empty(shape=[len(lftstept), 1], dtype=int)
        i = 0
        for x in lftstept:
            indexlstep[i] = np.argmin(np.abs(x - t))  # find ftsteptimes index location
            i = i+1
        data.stable.iloc[np.unique(indexlstep)] = 2
    if len(rftstept) !=0:
        # label here right foot step times
        indexrstep = np.empty(shape=[len(rftstept), 1], dtype=int)
        i = 0
        for x in rftstept:
            indexrstep[i] = np.argmin(np.abs(x - t))
            i = i + 1
        data.stable.iloc[np.unique(indexrstep)] = 2
   # plt.plot(t, data.stable)
    return data

##########################################################################
def label_pert(data, pertinfo, t):
    """ find index times of when pert started and ended and label data as 1

    :param t: time of data in same frame rate
    :param data: emg or mrk data
    :param pertinfo: dataframe with: ('pertStart', 'pertEnd', 'pertType', 'pertValue', 'pertDirt',
                                          'pertampVal', 'pertlvl', 'heightwhenpert')
    :return:
    """
    i = 0
    out = pertinfo.values
    visualp = np.array(np.where(out[:, 2] == 2))
    out = np.delete(out, visualp, axis=0)
    for x in out[:, 0]:
        indexstart = np.argmin(np.abs(x - t))
        indexend = np.argmin(np.abs(out[i, 1] - t))
        i = i + 1
        data.stable.iloc[indexstart:indexend] = 1
    return data