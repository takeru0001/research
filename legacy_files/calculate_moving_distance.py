import gc
import os
from math import acos, cos, radians, sin

import matplotlib.pyplot as plt

parking_lots = [[-122.39650143, -122.39259971], [37.74989626, 37.75248914]]
earth_rad = 6378.137


def latlng_to_xyz(lat, lng):
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat * cos(rlng), coslat * sin(rlng), sin(rlat)


def dist_on_sphere(pos0, pos1, radius=earth_rad):
    if pos0 == pos1:
        return 0
    xyz0, xyz1 = latlng_to_xyz(*pos0), latlng_to_xyz(*pos1)
    return acos(sum(x * y for x, y in zip(xyz0, xyz1, strict=False))) * radius


filelist = "./cabspottingdata/_cabs.txt"
infilename_list = []
with open(filelist) as f:
    for line in f:
        line = line.rstrip().split('"')
        filepath = "./cabspottingdata/" + "new_" + line[1] + ".txt"
        infilename_list.append(filepath)


# filelist = "turnover_rates.txt"
# infilename_list = []
# with open(filelist,"r") as f:
#    for line in f:
#        line = line.rstrip().split(":")
#        filepath = "./cabspottingdata/" + "new_" + line[0] + ".txt"
#        infilename_list.append(filepath)


if not os.path.exists("./taxi"):
    os.makedirs("taxi")

distance_transition_boarding_list = []
distance_transition_dropoff_list = []
time_transition_boarding_list = []
time_transition_dropoff_list = []

cnt = 0
fig = plt.figure()
# fig.patch.set_alpha(0)
ax1 = fig.add_subplot(211, xlim=(0, 120), ylim=(0, 50))
ax2 = fig.add_subplot(212, xlim=(0, 120), ylim=(0, 50))
# ax1.patch.set_alpha(0)
# ax2 = fig.add_subplot(2,1,2)

for infilename in infilename_list:
    # for infilename in infilename_list[:10]: #top10
    # for infilename in infilename_list[-10:]: #bottom10
    cnt += 1
    print(infilename + " " + str(cnt))
    distance_transition_boarding_list = []
    distance_transition_dropoff_list = []
    time_transition_boarding_list = []
    time_transition_dropoff_list = []
    with open(infilename) as f:
        data_list = []
        for line in f:
            data_tmp_list = line.rstrip("\n").split()
            unixtime = int(data_tmp_list[3])
            longitude = float(data_tmp_list[1])
            latitude = float(data_tmp_list[0])
            ride_state = int(data_tmp_list[2])
            data = [unixtime, longitude, latitude, ride_state]
            data_list.append(data)
        data_list = sorted(data_list)

    is_first_line = True
    prev_longitude = None
    prev_latitude = None
    prev_coordinate = None
    prev_ride_state = None
    prev_unixtime = None
    start_time_point = None
    start_coordinate = None
    latest_normal_timepoint = None
    latest_normal_coordinate = None
    latest_normal_ride_state = None
    distance_transition_boarding = []
    distance_transition_dropoff = []
    time_transition_boarding = []
    time_transition_dropoff = []

    for data in data_list:
        unixtime = data[0]
        longitude = data[1]
        latitude = data[2]
        ride_state = data[3]
        coordinate = latitude, longitude
        if is_first_line:
            prev_longitude = longitude
            prev_latitude = latitude
            prev_coordinate = latitude, longitude
            prev_ride_state = ride_state
            prev_unixtime = unixtime
            start_time_point = unixtime
            start_coordinate = latitude, longitude
            is_first_line = False
            continue

        # 空車かつタクシー会社の駐車場に停車(時速10km以下)している場合は除外する / ride_stateを2にする
        if ride_state == 0:
            if (
                parking_lots[0][0] < longitude < parking_lots[0][1]
                and parking_lots[1][0] < latitude < parking_lots[1][1]
            ):
                distance = dist_on_sphere(prev_coordinate, coordinate)
                elapsed_hours = (unixtime - prev_unixtime) / 3600
                hourly_speed = distance / elapsed_hours
                if hourly_speed < 10:
                    ride_state = 2

        # データの時間間隔がn秒以上空いた場合,データを区切る
        n = 600
        if unixtime - prev_unixtime > n:
            if prev_ride_state == 0 and len(distance_transition_dropoff) != 0:
                distance_transition_dropoff_list.append(distance_transition_dropoff)
                distance_transition_dropoff = []
                time_transition_dropoff_list.append(time_transition_dropoff)
                time_transition_dropoff = []
            elif prev_ride_state == 1 and len(distance_transition_boarding) != 0:
                distance_transition_boarding_list.append(distance_transition_boarding)
                distance_transition_boarding = []
                time_transition_boarding_list.append(time_transition_boarding)
                time_transition_boarding = []
            start_time_point = unixtime
            start_coordinate = coordinate
            prev_longitude = longitude
            prev_latitude = latitude
            prev_coordinate = latitude, longitude
            prev_ride_state = ride_state
            prev_unixtime = unixtime
            continue

        # 時速140km以上の場合、位置情報が正常に取得できていないと判断し、そのエラーデータをパスする
        if (
            dist_on_sphere(prev_coordinate, coordinate)
            / ((unixtime - prev_unixtime) / 3600)
            > 140
            and latest_normal_timepoint == None
        ):
            latest_normal_timepoint = prev_unixtime
            latest_normal_coordinate = prev_coordinate
            latest_normal_ride_state = prev_ride_state
            prev_longitude = longitude
            prev_latitude = latitude
            prev_coordinate = latitude, longitude
            prev_ride_state = ride_state
            prev_unixtime = unixtime
            continue

        # 1つ前のデータがエラーデータの場合、最後の正常なデータと今のデータ間の時速が140km以上であれば、今のデータもエラーデータとして扱う
        # そうでなければ再度正常なデータとして扱う
        if latest_normal_timepoint != None:
            if (
                dist_on_sphere(latest_normal_coordinate, coordinate)
                / ((unixtime - latest_normal_timepoint) / 3600)
                > 140
            ):
                prev_longitude = longitude
                prev_latitude = latitude
                prev_coordinate = latitude, longitude
                prev_ride_state = ride_state
                prev_unixtime = unixtime
                continue
            else:
                latest_normal_timepoint = None
                latest_normal_coordinate = None
                # エラーデータを跨いで、ride_stateが変わっていた場合はデータを区切る
                if latest_normal_ride_state != ride_state:
                    if (
                        latest_normal_ride_state == 0
                        and len(distance_transition_dropoff) != 0
                    ):
                        distance_transition_dropoff_list.append(
                            distance_transition_dropoff
                        )
                        distance_transition_dropoff = []
                        time_transition_dropoff_list.append(time_transition_dropoff)
                        time_transition_dropoff = []
                    elif (
                        latest_normal_ride_state == 1
                        and len(distance_transition_boarding) != 0
                    ):
                        distance_transition_boarding_list.append(
                            distance_transition_boarding
                        )
                        distance_transition_boarding = []
                        time_transition_boarding_list.append(time_transition_boarding)
                        time_transition_boarding = []
                    start_time_point = unixtime
                    start_coordinate = coordinate
                latest_normal_ride_state = None

        if prev_ride_state == 1:
            # elapsed_minutes = (unixtime - start_time_point + 30) // 60
            elapsed_time = (unixtime - start_time_point) / 60
            time_transition_boarding.append(elapsed_time)
            distance_transition_boarding.append(
                dist_on_sphere(start_coordinate, coordinate)
            )
            if ride_state == 0 or ride_state == 2:
                distance_transition_boarding_list.append(distance_transition_boarding)
                distance_transition_boarding = []
                time_transition_boarding_list.append(time_transition_boarding)
                time_transition_boarding = []
                start_time_point = unixtime
                start_coordinate = coordinate

        if prev_ride_state == 0:
            elapsed_time = (unixtime - start_time_point) / 60
            time_transition_dropoff.append(elapsed_time)
            distance_transition_dropoff.append(
                dist_on_sphere(start_coordinate, coordinate)
            )
            if ride_state == 1 or ride_state == 2:
                distance_transition_dropoff_list.append(distance_transition_dropoff)
                distance_transition_dropoff = []
                time_transition_dropoff_list.append(time_transition_dropoff)
                time_transition_dropoff = []
                start_time_point = unixtime
                start_coordinate = coordinate

        if prev_ride_state == 2:
            if ride_state != 2:
                start_time_point = unixtime
                start_coordinate = coordinate

        prev_longitude = longitude
        prev_latitude = latitude
        prev_coordinate = latitude, longitude
        prev_ride_state = ride_state
        prev_unixtime = unixtime

    if len(distance_transition_boarding) != 0:
        distance_transition_boarding_list.append(distance_transition_boarding)
        time_transition_boarding_list.append(time_transition_boarding)
        distance_transition_boarding = []
        time_transition_boarding = []

    if len(distance_transition_dropoff) != 0:
        distance_transition_dropoff_list.append(distance_transition_dropoff)
        time_transition_dropoff_list.append(time_transition_dropoff)
        distance_transition_dropoff = []
        time_transition_dropoff = []

    for i in range(len(distance_transition_boarding_list)):
        # if i % 100 == 0:
        #    print(i)
        x = time_transition_boarding_list[i]
        y = distance_transition_boarding_list[i]
        # if max(y) > 50 or max(x) > 120:
        #    continue
        ax1.plot(x, y, linewidth=0.05)

    for i in range(len(distance_transition_dropoff_list)):
        x = time_transition_dropoff_list[i]
        y = distance_transition_dropoff_list[i]
        # if max(y) > 50 or max(x) > 300:
        # if max(y) > 50 or max(x) > 120:
        #    continue
        ax2.plot(x, y, linewidth=0.05)

    print(len(distance_transition_boarding_list))
    print(len(distance_transition_dropoff_list))
    del time_transition_boarding_list
    del distance_transition_boarding_list
    del time_transition_dropoff_list
    del distance_transition_dropoff_list
    gc.collect()

    # if cnt == 10:
    #    break

ax1.set_xlabel("time[minute]")
ax1.set_ylabel("from boarding point[km]")
ax2.set_xlabel("time[minutes]")
ax2.set_ylabel("from drop off point[km]")

# plt.savefig("taxi/boarding/" + infilename[22:-4] + "_boarding")
# plt.savefig("taxi/" + "all_taxis" + str(cnt))
# plt.savefig("taxi/" + "top10")
ax1.set_title("top10")
plt.savefig("taxi/" + "top10")
plt.cla()
plt.clf()
plt.close()
