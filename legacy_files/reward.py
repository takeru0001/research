from math import sin, cos, acos, radians
earth_rad = 6378.137

def latlng_to_xyz(lat, lng):
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat*cos(rlng), coslat*sin(rlng), sin(rlat)


def dist_on_sphere(pos0_latitude, pos0_longitude, pos1_latitude, pos1_longitude, radius=earth_rad):
    pos0 = pos0_latitude, pos0_longitude
    pos1 = pos1_latitude, pos1_longitude
    if pos0 == pos1:
        return 0
    xyz0, xyz1 = latlng_to_xyz(*pos0), latlng_to_xyz(*pos1)
    return acos(sum(x * y for x, y in zip(xyz0, xyz1)))*radius


def speed_calculation(unixtime_difference, moving_distance):
    elapsed_hour = unixtime_difference / 3600
    if elapsed_hour == 0:
        hourly_speed = 0
    else:
        hourly_speed = moving_distance / elapsed_hour
    return hourly_speed


def reward_calculation(coordinates_in_ride):
    #料金、出発、到着地を返す
    #最高時速が140kmを超える場合には、エラーデータとしてrewardをNoneで返す

    mile = 1.60934
    base_fare = 3.50 # first 0.2 mile
    additional_fare = 0.55 # per 0.2 mile
    waiting_fare = 0.55 # per a minute

    orig = coordinates_in_ride[0]["latitude"], coordinates_in_ride[0]["longitude"]
    dist = coordinates_in_ride[-1]["latitude"], coordinates_in_ride[-1]["longitude"]

    prev_data_dict = None
    reward = base_fare
    for data_dict in coordinates_in_ride:
        if prev_data_dict is None:
            prev_data_dict = data_dict
            continue
        moving_distance = dist_on_sphere(prev_data_dict["latitude"], prev_data_dict["longitude"], data_dict["latitude"], data_dict["longitude"])
        unixtime_difference = data_dict["unixtime"] - prev_data_dict["unixtime"]
        hourly_speed = speed_calculation(unixtime_difference, moving_distance)
        if hourly_speed > 140:
            reward = None
            #print("error 140km")
            return reward, orig, dist
        elif hourly_speed < 10:
            reward += (unixtime_difference / 60) * waiting_fare
        else:
            moving_mile = moving_distance / mile
            reward += (moving_mile / 0.2) * additional_fare
        prev_data_dict = data_dict

    return reward, orig, dist

