import os
import shutil

filelist = "./cabspottingdata/_cabs.txt"
infilename_list = []
with open(filelist) as f:
    for line in f:
        line = line.rstrip().split('"')
        filepath = "./cabspottingdata/" + "new_" + line[1] + ".txt"
        infilename_list.append(filepath)

if not os.path.exists("./cabspottingdata_returner_airport"):
    os.makedirs("cabspottingdata_returner_airport")

new_infilename_list = []


# infilename_list = ["./cabspottingdata/new_abboip.txt"]
for infilename in infilename_list:
    infile = open(infilename)
    airport_count = 0
    down_town_count = 0
    for line in infile:
        data_tmp_list = line.split(" ")
        unixtime = data_tmp_list[3].replace("\r\n", "")
        longitude = float(data_tmp_list[1])
        latitude = float(data_tmp_list[0])
        if 37.650508 > latitude > 37.592989 and -122.446575 < longitude < -122.348721:
            airport_count += 1
        # if (37.708441 > latitude > 37.574652 and -122.525680< longitude < -122.337455):
        #   down_town_count += 1
    # print(airport_count)
    if airport_count > 3500:
        convert_file_name = infilename.split("/")
        newFileName = "./cabspottingdata_returner_airport/" + convert_file_name[2]
        new_infilename_list.append(newFileName)
        shutil.copy2(infilename, newFileName)

    infile.close()

with open("./cabspottingdata_returner_airport/_cabs.txt", "w") as f:
    for file_name in new_infilename_list:
        f.write(str(file_name) + "\n")
