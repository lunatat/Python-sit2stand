import pandas
import tools.vicon as vn
import numpy as np

filename = vn.fxn_select_files()  # Ask user to select file to load
data = pandas.read_excel(filename[0])

datatest = data
datalvl1 = data.loc[data['pertlvl'] == 1]
datalvl2 = data.loc[data['pertlvl'] == 2]
datalvl3 = data.loc[data['pertlvl'].isin([3])]
subjects = pandas.DataFrame(data=data.subject.unique(), columns={'subject'})

count = 0
for i in [0, 1]:
    for k in [0, 4]:
        lv1 = datalvl1.loc[datalvl1['pertType'] == i].loc[datalvl1['pertDirt'] == k].add_suffix('_1')
        lv2 = datalvl2.loc[datalvl2['pertType'] == i].loc[datalvl2['pertDirt'] == k].add_suffix('_2')
        lv3 = datalvl3.loc[datalvl3['pertType'] == i].loc[datalvl3['pertDirt'] == k].add_suffix('_3')
        datawide = lv1.set_index('subject_1').join(lv2.set_index('subject_2')).join(lv3.set_index('subject_3'))

        if count == 0:
            dataXwide = datawide
        else:
            dataXwide = pandas.concat(
                [dataXwide, datawide], axis=0
            )
        count = count + 1


# convert nans to 9999
dataXwide = dataXwide.fillna(9999)
dataXwide = dataXwide.reset_index()
dataXwide.to_excel('datawide-4-7-2022.xlsx')
