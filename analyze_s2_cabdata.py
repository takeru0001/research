#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import shutil

effective_digits = 2

filelist = "./cabspottingdata/_cabs.txt"
infilename_list = []
with open(filelist,"r") as f:
  for line in f:
    line = line.rstrip().split('"')
    filepath = "./cabspottingdata/" + "new_" + line[1] + ".txt"
    infilename_list.append(filepath)

if not os.path.exists("./cabspottingdata_returner"):
  os.makedirs("cabspottingdata_returner")

prev_ride_state = 0
new_infilename_list = []

for infilename in infilename_list:
  try:
    traces_list = []
    infile = open(infilename, "r")
    for line in infile:
      data_tmp_list = line.split(" ")
      unixtime = int(data_tmp_list[3].replace('\r\n',''))
      longitude = float(data_tmp_list[1])
      latitude = float(data_tmp_list[0])
      ride_state = int(data_tmp_list[2])
      if (prev_ride_state == 1 and ride_state == 0):
        traces_list.append( (round(latitude, effective_digits), round(longitude, effective_digits)) )
      prev_ride_state = ride_state


    # print(traces_list[0][0])
    # 重心計算
    cm_lat = 0
    cm_lon = 0
    r_cm = []
    for i in range(len(traces_list)):
      cm_lat += traces_list[i][0]
      cm_lon += traces_list[i][1]
    cm_lat = cm_lat / float(len(traces_list))
    cm_lon = cm_lon / float(len(traces_list))
    r_cm = [cm_lat, cm_lon]

    # 旋回半径の抽出
    R_g = 0.0
    for i in range(len(traces_list)):
      R_g += (traces_list[i][0] - r_cm[0])**2 + (traces_list[i][1] - r_cm[1])**2
    R_g = np.sqrt(R_g / float(len(traces_list)))

    # k(=2)-旋回半径（移動半径） r_g^{(k)} の計算
    frequency_dict = {}
    sorted_frequency_dict = []
    for i in range(len(traces_list)):
      if traces_list[i] not in frequency_dict.keys():
        frequency_dict[ traces_list[i] ] = 1
      else:
        frequency_dict[ traces_list[i] ] += 1
    sorted_frequency_dict = sorted(frequency_dict.items(), key=lambda x:x[1], reverse=True)
    # print(sorted_frequency_dict, sorted_frequency_dict[0], sorted_frequency_dict[1])

    R_g2_cm_lat = 0
    R_g2_cm_lon = 0
    R_g2_cm_lat = (sorted_frequency_dict[0][1] * sorted_frequency_dict[0][0][0] + sorted_frequency_dict[1][1] * sorted_frequency_dict[1][0][0]) / (sorted_frequency_dict[0][1] + sorted_frequency_dict[1][1])
    R_g2_cm_lon = (sorted_frequency_dict[0][1] * sorted_frequency_dict[0][0][1] + sorted_frequency_dict[1][1] * sorted_frequency_dict[1][0][1]) / (sorted_frequency_dict[0][1] + sorted_frequency_dict[1][1])

    # r_g^{(2)}の計算
    R_g2 = 0
    R_g2 = sorted_frequency_dict[0][1] * ( (sorted_frequency_dict[0][0][0] - R_g2_cm_lat)**2 + (sorted_frequency_dict[0][0][1] - R_g2_cm_lon)**2 ) + sorted_frequency_dict[1][1] * ( (sorted_frequency_dict[1][0][0] - R_g2_cm_lat)**2 + (sorted_frequency_dict[1][0][1] - R_g2_cm_lon)**2 ) 
    R_g2 = np.sqrt(R_g2/float(sorted_frequency_dict[0][1]+sorted_frequency_dict[1][1]))

    s2 = 0
    s2 = R_g2/R_g

    if(s2 > 0.9):
      convert_file_name = infilename.split('/')
      newFileName = "./cabspottingdata_returner/" + convert_file_name[2]
      new_infilename_list.append(newFileName)
      shutil.copy2(infilename, newFileName)


    infile.close()

  except:
    print(infilename)

with open("./cabspottingdata_returner/_cabs.txt", "w") as f:
  for file_name in new_infilename_list:
    f.write(str(file_name) + "\n")

