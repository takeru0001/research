# coding: utf-8
from operator import itemgetter
from math import sin, cos, acos, radians
earth_rad = 6378.137

def latlng_to_xyz(lat, lng):
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat*cos(rlng), coslat*sin(rlng), sin(rlat)

def dist_on_sphere(pos0, pos1, radius=earth_rad):
    if pos0 == pos1:
        return 0
    xyz0, xyz1 = latlng_to_xyz(*pos0), latlng_to_xyz(*pos1)
    return acos(sum(x * y for x, y in zip(xyz0, xyz1)))*radius


filelist = "./cabspottingdata/_cabs.txt"
infilename_list = []
with open(filelist,"r") as f:
    for line in f:
        line = line.rstrip().split('"')
        filepath = "./cabspottingdata/" + "new_" + line[1] + ".txt"
        infilename_list.append(filepath)

#infilename_list = ["./cabspottingdata/new_ancorjo.txt"]
#infilename_list = ["./cabspottingdata/new_amwibs.txt"]

mile = 1.60934
base_fare = 3.50 # first 0.2 miles
additional_fare = 0.55 # per 0.2 miles
waiting_fare = 0.55 # per a minute

turnover_rate_list = []
for infilename in infilename_list:
    print(infilename)
    with open(infilename,"r") as f:
        data_list = []
        for line in f:
            data_tmp_list = line.split(" ")
            unixtime = int(data_tmp_list[3].replace('\r\n',''))
            longitude = float(data_tmp_list[1])
            latitude = float(data_tmp_list[0])
            ride_state = int(data_tmp_list[2])
            data = [unixtime, longitude, latitude, ride_state]
            data_list.append(data)
        data_list = sorted(data_list)

        is_first_line = True
        prev_ride_state = None
        prev_unixtime = None
        prev_coordinate = None
        biginning_time_point = None
        empty_time = 0
        riding_time = 0
        moving_distance = 0
        biginning_coordinate = None
        waiting_minutes = 0
        tmp_fares = 0
        fares = 0
        parking_lots = [[-122.39650143, -122.39259971], [37.74989626, 37.75248914]]
        cnt = 0
        for data in data_list:
            unixtime = data[0]
            longitude = data[1]
            latitude = data[2]
            coordinate = latitude, longitude
            ride_state = data[3] #0==空車, 1==実車
            #print(cnt)
            #print(coordinate)
            #print(prev_coordinate)

            if is_first_line == True:
                prev_coordinate = latitude, longitude
                prev_ride_state = ride_state
                prev_unixtime = unixtime
                biginning_time_point = unixtime
                biginning_coordinate = latitude, longitude
                is_first_line = False
                continue

            #データの時間間隔がn秒以上空いた場合、その間の時間を計算から除外する
            #また、それまで乗客が乗っていた場合、乗客を降ろしたものとして処理する
            n = 600
            if unixtime - prev_unixtime > n:
                if prev_ride_state == 0:
                    empty_time += prev_unixtime - biginning_time_point
                elif prev_ride_state == 1:
                    riding_time += prev_unixtime - biginning_time_point
                    distance = dist_on_sphere(biginning_coordinate, prev_coordinate)
                    moving_distance += distance
                    elapsed_hours = (prev_unixtime - biginning_time_point) / 3600
                    if elapsed_hours == 0:
                        hourly_speed = 0
                    else:
                        hourly_speed = distance / elapsed_hours
                    #乗車時間が3分未満の場合、平均時速が100kmを超える場合はエラーデータとして料金にカウントしない
                    if prev_unixtime - biginning_time_point > 180 and hourly_speed < 100:
                        if dist_on_sphere(biginning_coordinate, prev_coordinate) / mile < 0.2:
                            fares += base_fare + (waiting_minutes * waiting_fare)
                        else:
                            fares += tmp_fares + (base_fare - additional_fare) + (waiting_minutes * waiting_fare)
                biginning_time_point = unixtime
                biginning_coordinate = coordinate
                prev_coordinate = latitude, longitude
                prev_ride_state = ride_state
                prev_unixtime = unixtime
                tmp_fares = 0
                waiting_minutes = 0
                continue

            #タクシー会社の駐車場に停車(時速10km以下)している場合は、計算から除外する
            if parking_lots[0][0] < longitude < parking_lots[0][1] and parking_lots[1][0] < latitude < parking_lots[1][1]:
                distance = dist_on_sphere(prev_coordinate, coordinate)
                elapsed_hours = (unixtime - prev_unixtime) / 3600
                hourly_speed = distance / elapsed_hours

                if hourly_speed < 10:
                    ride_state = 2
                    if prev_ride_state == 0:
                        empty_time += unixtime - biginning_time_point
                    if prev_ride_state == 1:
                        riding_time += unixtime - biginning_time_point
                        if biginning_coordinate != coordinate:
                            moving_distance += dist_on_sphere(biginning_coordinate, coordinate)
                    biginning_time_point = None

            #タクシー会社の駐車場から出た場合
            if prev_ride_state == 2 and ride_state != 2:
                biginning_time_point = unixtime
                biginning_coordinate = coordinate

            if prev_ride_state == 0 and ride_state == 1:
                empty_time += unixtime - biginning_time_point
                biginning_time_point = unixtime
                biginning_coordinate = coordinate

            if prev_ride_state == 1:
                distance = dist_on_sphere(prev_coordinate, coordinate)
                elapsed_hours = (unixtime - prev_unixtime) / 3600
                hourly_speed = distance / elapsed_hours
                if hourly_speed < 10: #時速10km以下の場合、待機料金で計算
                    waiting_minutes += (elapsed_hours * 60)
                elif hourly_speed > 140: #時速140km以上の場合、位置情報が正常に取得できていないと判断し、料金を加算しない
                    tmp_fares += 0
                else:
                    tmp_fares += ((distance / mile) / 0.2) * additional_fare

                if ride_state == 0:
                    riding_time += unixtime - biginning_time_point
                    moving_distance += dist_on_sphere(biginning_coordinate, coordinate)
                    distance = dist_on_sphere(biginning_coordinate, coordinate)
                    elapsed_hours = (unixtime - biginning_time_point) / 3600
                    hourly_speed = distance / elapsed_hours

                    #乗車時間が3分未満の場合、平均時速が100kmを超える場合はエラーデータとして料金にカウントしない
                    if unixtime - biginning_time_point > 180 and hourly_speed < 100:
                        if dist_on_sphere(biginning_coordinate, coordinate) * mile < 0.2:
                            fares += base_fare + (waiting_minutes / waiting_fare)
                        else:
                            fares += tmp_fares + (base_fare - additional_fare) + (waiting_minutes * waiting_fare)

                    biginning_time_point = unixtime
                    biginning_coordinate = coordinate
                    tmp_fares = 0
                    waiting_minutes = 0

            prev_coordinate = latitude, longitude
            prev_ride_state = ride_state
            prev_unixtime = unixtime

            cnt += 1

        if ride_state == 0:
            empty_time += unixtime - biginning_time_point

        if ride_state == 1:
            riding_time += unixtime - biginning_time_point
            moving_distance += dist_on_sphere(biginning_coordinate, coordinate)
            fares += tmp_fares

        #データ数が極端に少ないファイルを除外
        if cnt > 5000:
            #turnover_rate = moving_distance / ((empty_time + riding_time) / 3600)
            turnover_rate = fares / ((empty_time + riding_time) / 3600)
            turnover_rate_list.append([infilename[22:-4], turnover_rate])

        #print(moving_distance)
        #print(empty_time)
        #print(riding_time)

turnover_rate_list = sorted(turnover_rate_list, key=itemgetter(1), reverse=True)
with open("turnover_rates.txt","w",encoding="utf-8") as f:
    for turnover_rate in turnover_rate_list:
        f.write(":".join(map(str,turnover_rate)) + "\n")


