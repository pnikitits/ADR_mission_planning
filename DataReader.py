import scipy.io
from Debris import Debris

mat_data = scipy.io.loadmat('Data/TLE_iridium.mat')


variable_names = mat_data.keys()
print("Variable Names:")
for name in variable_names:
    print(name)

    
tle_iridium_data = mat_data['TLE_iridium']


norad_list = tle_iridium_data[0, :]
inclination_list = tle_iridium_data[1, :]
raan_list = tle_iridium_data[2, :]
eccentricity_list = tle_iridium_data[3, :]
arg_perigee_list = tle_iridium_data[4, :]
mean_anomaly_list = tle_iridium_data[5, :]
a_list = tle_iridium_data[6, :]
rcs_list = tle_iridium_data[7, :]


def make_Iridium_debris():
    output = []
    for i in range(0 , len(norad_list)-1):
        norad = norad_list[i]
        inclination = inclination_list[i]
        raan = raan_list[i]
        eccentricity = eccentricity_list[i]
        arg_perigee = arg_perigee_list[i]
        mean_anomaly = mean_anomaly_list[i]
        a = a_list[i]/6371 # normalise the semi major axis

        debris = Debris(norad=norad,
                        inclination=inclination,
                        raan=raan,
                        eccentricity=eccentricity,
                        arg_perigee=arg_perigee,
                        mean_anomaly=mean_anomaly,
                        a=a,
                        rcs=None)
        output.append(debris)

    return output