#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 分析用データ
traces_list = []
max_lat = -90.0
min_lat = 90.0
max_lon = -180.0
min_lon = 180.0
effective_digits = 2
print("緯度・経度の有効数字:", effective_digits)
#time_to_analyze = 60000
time_to_analyze = 30000
#time_to_analyze = 20000

# main

# データファイルの読み込み
#infilename = "destination_coordinates_data0.8.txt"
#infilename = "analyze_rg-cars70-div40-epslion0.8.csv"
#infilename = "analyze_rg-cars70-div40-epslion0.0.csv"
#infilename = "analyze_rg-cars70-div40-epslion0.05.csv"
infilename = "destination_coordinates_data0.8.txt"
infile = open(infilename, "r")

for line in infile:
  # データファイルから属性値を抽出
  data_list = line.replace("\n", "").replace("(", "").replace(")", "").split(",")
  time = int(data_list[1])
  start_lat = float(data_list[2])
  start_lon = float(data_list[3])
  end_lat = float(data_list[4])
  end_lon = float(data_list[5])
  #print(time, start_lat, start_lon, end_lat, end_lon)
  #sys.exit(0)

  # 地図の緯度・経度の最小値・最大値の導出
  if start_lat > max_lat:
    max_lat = start_lat
  if end_lat > max_lat:
    max_lat = end_lat

  if start_lon > max_lon:
    max_lon = start_lon
  if end_lon > max_lon:
    max_lon = end_lon

  if start_lat < min_lat:
    min_lat = start_lat
  if end_lat < min_lat:
    min_lat = end_lat

  if start_lon < min_lon:
    min_lon = start_lon
  if end_lon < min_lon:
    min_lon = end_lon

  if time >= time_to_analyze:
  # 車毎のトレースを抽出（小数点以下第<effective_digits>位まで有効）
    new_trace_flag = True
    for i in range(len(traces_list)):
      if len(traces_list[i]) >= 2 and (round(start_lat, effective_digits), round(start_lon, effective_digits)) == traces_list[i][-1]:
        traces_list[i].append( (round(end_lat, effective_digits), round(end_lon, effective_digits)) )
        new_trace_flag = False
        break
    if new_trace_flag == True:
      traces_list.append( [ (round(start_lat, effective_digits), round(start_lon, effective_digits)), (round(end_lat, effective_digits), round(end_lon, effective_digits)) ] )


## トレースの出力
#for i in range(len(traces_list)):
#  print(traces_list[i])
#print(len(traces_list))
#sys.exit(0)

# トレースの抽出
lat_lon_list = []
for i in range(len(traces_list)):
  lat_list = []; lon_list = []
  for j in range(len(traces_list[i])):
    lat_list.append( traces_list[i][j][0] )
    lon_list.append( traces_list[i][j][1] )
  lat_lon_list.append( [lat_list, lon_list] )
#  print(lat_lon_list[i])
#  plt.plot(lon_list, lat_list, linestyle="", marker=".")
#  plt.show()
#  plt.clf()
#sys.exit(0)


# 各車のトレースの重心 r_{cm} の計算
# 参考 ( https://www.nature.com/articles/ncomms9166.pdf )
r_cm_list = []
for i in range(len(traces_list)):
  cm_lat = 0.0; cm_lon = 0.0
  for j in range(len(traces_list[i])):
    cm_lat += traces_list[i][j][0]
    cm_lon += traces_list[i][j][1]
  cm_lat = cm_lat / float(len(traces_list[i]))
  cm_lon = cm_lon / float(len(traces_list[i]))
  #print(i, cm_lat, cm_lon)
  r_cm_list.append( (cm_lat, cm_lon) )

# トレースの重心 r_{cm} の出力
number_of_cars = len(r_cm_list)
print("車の台数:", number_of_cars)
print("各車のトレースの重心 r_{cm}:", r_cm_list)
#sys.exit(0)


# 旋回半径(移動半径) r_g の計算
# 参考 ( https://www.nature.com/articles/ncomms9166.pdf )
R_g_list = []
for i in range(len(traces_list)):
  R_g = 0.0
  for j in range(len(traces_list[i])):
    R_g += (traces_list[i][j][0] - r_cm_list[i][0])**2 + (traces_list[i][j][1] - r_cm_list[i][1])**2
  R_g = np.sqrt(R_g / float(len(traces_list[i])))
  R_g_list.append( R_g )


# 旋回半径(移動半径) R_g の出力
print("旋回半径 r_g", R_g_list)
#sys.exit(0)


# k(=2)-旋回半径（移動半径） r_g^{(k)} の計算
# 参考: 
# https://www.nature.com/articles/ncomms9166.pdf 
# https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9166/MediaObjects/41467_2015_BFncomms9166_MOESM1165_ESM.pdf
R_g2_list = []
for i in range(len(traces_list)):
  # 訪問頻度を計算
  frequency_dict = {}
  for j in range(len(traces_list[i])):
    if traces_list[i][j] not in frequency_dict.keys():
      frequency_dict[ traces_list[i][j] ] = 1
    else:
      frequency_dict[ traces_list[i][j] ] += 1
  sorted_frequency_dict = sorted(frequency_dict.items(), key=lambda x:x[1], reverse=True)
  #print(sorted_frequency_dict, sorted_frequency_dict[0], sorted_frequency_dict[1])
  print("車"+str(i)+"の訪問頻度1位:", sorted_frequency_dict[0])
  print("車"+str(i)+"の訪問頻度2位:", sorted_frequency_dict[1])

  # トレースの描画
  plt.plot(lat_lon_list[i][1], lat_lon_list[i][0], linestyle="", marker=".", color="black")
  plt.plot([sorted_frequency_dict[0][0][1]], [sorted_frequency_dict[0][0][0]], marker="o", color="red")
  plt.plot([sorted_frequency_dict[1][0][1]], [sorted_frequency_dict[1][0][0]], marker="o", color="blue")
  plt.show()
  plt.clf()
  #sys.exit(0)


  R_g2_cm_lat = (sorted_frequency_dict[0][1] * sorted_frequency_dict[0][0][0] + sorted_frequency_dict[1][1] * sorted_frequency_dict[1][0][0]) / (sorted_frequency_dict[0][1] + sorted_frequency_dict[1][1])
  R_g2_cm_lon = (sorted_frequency_dict[0][1] * sorted_frequency_dict[0][0][1] + sorted_frequency_dict[1][1] * sorted_frequency_dict[1][0][1]) / (sorted_frequency_dict[0][1] + sorted_frequency_dict[1][1])
  print("車"+str(i)+"のr_{cm}^{(k)} =", (R_g2_cm_lat, R_g2_cm_lon))

  # r_g^{(2)}の計算
  R_g2 = sorted_frequency_dict[0][1] * ( (sorted_frequency_dict[0][0][0] - R_g2_cm_lat)**2 + (sorted_frequency_dict[0][0][1] - R_g2_cm_lon)**2 ) + sorted_frequency_dict[1][1] * ( (sorted_frequency_dict[1][0][0] - R_g2_cm_lat)**2 + (sorted_frequency_dict[1][0][1] - R_g2_cm_lon)**2 ) 
  R_g2 = np.sqrt(R_g2/float(sorted_frequency_dict[0][1]+sorted_frequency_dict[1][1]))
  R_g2_list.append( R_g2 )

# k(=2)-旋回半径（移動半径） r_g^{(k)} の出力
print("k(=2)-旋回半径 r_g^{(2)}:", R_g2_list)

# s_2 の出力（最終結果）
s2_list = []
print("--- 結果（ここから）---")
for i in range(len(R_g_list)):
  s2_list.append( R_g2_list[i]/R_g_list[i] )
  print("車"+str(i)+"のs_2:", R_g2_list[i]/R_g_list[i])
print("--- 結果（ここまで）---")

# plot s_2
plt.hist(s2_list)
plt.xlabel("s_2")
plt.ylabel("frequency")
#plt.show()
plt.savefig("s_2-histgram.png")
