# Python-sit2stand
Sit-to-Stand-data-analysis

Hello user,

There are two main code files that you should refer to, they are under the tools folder.
1) vicon: there are functions in here you can implement in your code, such as, loading the vicon nexus csv files, filters you can use for force plates or EMG sensors etc. 
2) Analyze: Fxns to analyze sit to stand data, such as, segmenting sit to stand times (based on pelvic velocity and position data), 
  calculating joint angles (refer to documentation for rotation matrix/ and my paper to see what markers I used), symmetry index variables, margin of stability, and other output variables

- In the main folder are other scripts, like step01_load , this compiles all the data for the subject, calculates the output vars and saves it as a csv for stats later/graphing

-In the documentation folder you can find refrence materials for some of the variables calculated ex. coordinate frames for rotation matrices, margin of stability, etc.

Citation:
This is the code I used to analyze sit2stand data for the RA-L paper published:
doi: 10.1109/LRA.2022.3181351.
