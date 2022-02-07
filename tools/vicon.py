import os
import tkinter as tk
from tkinter import filedialog as fd
import numpy as np
import pandas
from typing import Tuple
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt


# Fxns to 1. import/read vicon and PXI files, 2. filter marker, force plate, and EMG data

def fxn_select_files() -> Tuple:
    """
    Asks user to select csv vicon file(s)

    :return: filename(s) as tuple
    """
    # ask user to select multiple files:
    root = tk.Tk()
    filenames = fd.askopenfilenames(parent=root, title='Select csv files', initialdir=os.getcwd())
    root.destroy()
    return filenames


def getpath(filepath: str, subject: str, file: str) -> str:
    """
    This edits filepath to include the subject folder and file.csv
    :param filepath: emg, pxi or force plate directory/file path
    :param subject: sub00
    :param file: test condition
    :return: file path in form of original filepath/subject/file.csv
    """
    filepath = filepath + subject + '/' + file + '.csv'
    return filepath


#########################################################################################

def read_vicon_XYZ_file(filename: str) -> pandas.DataFrame:
    """
    This functions reads a vicon file with variables that contain x,y,z coordinates

    :param filename: The path to the csv file to read
    :return: pandas DataFrame
    """
    # data = pandas.read_csv(filename, header=[0, 1], skiprows=2, usecols=np.arange(2, 77).tolist())
    data = pandas.read_csv(filename, header=None, skiprows=2, engine='python')
    data.dropna(axis='columns', how='all', inplace=True)  # deletes any columns that have entire NAs
    fq = pandas.read_csv(filename, nrows=1)  # only gets 2nd row where fq is saved
    out = data.values[3:, 2:]
    # converting object array to float: np.vstack(out[:, :]).astype(np.float)
    filtdata = filter_this(np.vstack(out[:, :]).astype(np.float), [fq.to_numpy(), 4, 10])
    mrknames = data.iloc[0, 2:].fillna(method='ffill')
    data = pandas.DataFrame(data=filtdata,
                            columns=(m + xyz for m in mrknames[:-1:3] for xyz in ('_x', '_y', '_z')))
    return data


def read_emg_csv(filename: str) -> pandas.DataFrame:
    """
    read vicon emg csv - only saves columns 10:16, 6 muscles
    :param filename: emg file path csv.
    :return: pandas dataframe of filtered emg information
    """
    muscles = ('lowerbck', 'abs', 'rectus_fem', 'bicep_fem', 'medial_gast', 'tibialis_ant')
    data = pandas.read_csv(filename, header=None, skiprows=5, engine='python')
    data.dropna(axis='columns', how='all', inplace=True)  # deletes any columns that have entire NAs
    data = pandas.DataFrame(data=data.values[:, 10:16], columns=muscles)
    fq = pandas.read_csv(filename, nrows=1)  # only gets 2nd row where fq is saved FIX
    out = data.values
    out = np.vstack(out[:, :]).astype(np.float)
    filtdata = filter_this(np.absolute(out - np.mean(out, axis=0)), [fq.to_numpy(), 4, 20, 500])
    filtdata = pandas.DataFrame(data=filtdata, columns=muscles)
    # quickplot(out[:, 0])
    # quickplot(filtdata.values[:, 0])
    return filtdata


def read_pxi_txt(filename: str) -> pandas.DataFrame:
    """
    read pxi output txt file
    :param filename: pxi output txt file
    :return: pandas dataframe of pxi information
    """
    pxivalues = ('time', 'framenum', 'dFx', 'dFy', 'dFz', 'dMx', 'dMy', 'dMz',
                 'pel_act_T1', 'pel_act_T2', 'pel_act_T3', 'pel_act_T4',
                 'pel_act_T5', 'pel_act_T6', 'pel_act_T7', 'pel_act_T8',
                 'pel_des_T1', 'pel_des_T2', 'pel_des_T3', 'pel_des_T4',
                 'pel_des_T5', 'pel_des_T6', 'pel_des_T7', 'pel_des_T8',
                 'qperror', 'aFx', 'aFy', 'aFz', 'aMx', 'aMy', 'aMz',
                 'applyPert', 'pertTrigger', 'pertDirection', 'pertOn',
                 'desTreadvelL', 'desTreadvelR', 'desTreadAccel',
                 'op2motor1', 'op2motor2', 'op2motor3', 'op2motor4',
                 'op2motor5', 'op2motor6', 'op2motor7', 'op2motor8',
                 'pertRT', 'PertCT', 'pertFT', 'PertampF', 'PertAccel', 'PertVisual',
                 'bothbelts', 'perttype', 'pertAtH')
    data = pandas.read_csv(filename, header=None, skiprows=7, engine='python', delimiter="\t")
    data = pandas.DataFrame(data=data.values[:, :], columns=pxivalues)
    return data


##################################################################################################

def filter_this(data, filtinfo):
    """
    Uses a butterworth filter then filtfilt

    :param data: float array data to filter
    :param filtinfo: array containing [frequency sampled,  order, freqcutoff, (optional)freqhighcutoff]
    :return: filteredD array with filtered data
    """
    filtered = np.empty(shape=data.shape, dtype=float)
    fs = filtinfo[0]
    fc = filtinfo[2:]
    wn = fc / (fs / 2)
    if len(fc) == 2:
        filtype = 'bandpass'
    else:
        filtype = 'lowpass'
    b, a = signal.butter(filtinfo[1], wn.flatten(), btype=filtype)  # this breaks when wn is [[lc hc]]-remove a []
    for i in range(0, len(data[0, :])):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i])
        if len(fc) == 2:
            fc2 = 3
            wn2 = fc2 / (fs / 2)
            b2, a2 = signal.butter(filtinfo[1], wn2.flatten(), btype='lowpass')
            filtered[:, i] = signal.filtfilt(b2, a2, np.absolute(filtered[:, i]))
    return filtered


##########################################################################

def quickplot(data):
    """

    :param data:
    :return:
    """
    sns.set_theme(style="whitegrid")  # gives it that pastel color
    ax = sns.lineplot(data=data)
    plt.show()

##########################################################################
