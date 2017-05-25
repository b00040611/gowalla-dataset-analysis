# -*- coding: utf-8 -*-
'''
The gowalla_dataset_user module defines:
    the Point class,
    the GowallaData class
    the GowallaUser class
'''
import numpy as np

import pandas as pd


from geopy.distance import vincenty

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import fclusterdata

from collections import OrderedDict

import pygmaps


def time_in_mins(x):
    return (x.hour)*60+x.minute

def CDF_plot(x):
    '''
    plot the CDF of a series in pandas dataframe
    '''
    ser = x.sort_values()
    #before proceeding, append again the last (and largest) value.
    #This step is important especially for small sample sizes in order to get an unbiased CDF:
    ser[len(ser)] = ser.iloc[-1]
    cum_dist = np.linspace(0., 1., len(ser))
    ser_cdf = pd.Series(cum_dist, index=ser)
#    plt.figure()
    ser_cdf.plot(drawstyle='steps')
    return True

def filter_night_checkin_data(dfc, lower_night_time_bound, upper_night_time_bound):
    """
    filter the check in data during the night.
    ：dfc: input data frame
    :lower_time_bound: end time of night period in the morning. e.g., 8am
    :upper_time_bound: start time of night period in the evening. e.g., 8pm
    ouput: dfc_night
    """
    dfc.loc[:, 'utc'] = dfc['utc'].astype('datetime64[ns]')
    dfc.loc[:, 'time'] = dfc['utc'].apply(lambda x: time_in_mins(x.time()))
    night = dfc[(dfc['time'] <= lower_night_time_bound) | (dfc['time'] >= upper_night_time_bound)]
    return night

def checkin_clustering(checkin, t, criterion, metric, method):
    '''
    Clustering the checkin data by distance.
    Using fclusterdata method in scipy
    Args:
        checkin: pandas frame having at least two columns ['lat','lon']
        t, criterion, metric, method are parameters of fclusterdata,
        see:https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html
    Returns:
        checkin_clusters: dict{clusterno:[(lat0,lon0),(lat1,lon1),...]}
        cluster_num_center:
            dict{clusterno:(number of points in the cluster, array([center_lat,center_lon]))}
    '''
    geopoint = np.array(checkin[['lat', 'lon']])
    clusters = fclusterdata(geopoint, t, criterion=criterion, metric=metric, depth=1, method=method)
    checkin_clusters = dict()
    index = 0
    for cluster_no in clusters:
        if cluster_no not in checkin_clusters.keys():
            checkin_clusters[cluster_no] = [tuple(geopoint[index])]
        else:
            checkin_clusters[cluster_no].append(tuple(geopoint[index]))
        index = index + 1
    cluster_number_center = dict()
    for cluster_no in clusters:
        cluster_number_center[cluster_no] = (len(checkin_clusters[cluster_no]), np.mean(np.array(checkin_clusters[cluster_no]), axis=0))
#   xlim =
#   ylim =
#   plt.scatter(geopoint[:,0], geopoint[:,1], c = clusters)
#   plt.show()
    checkin_clusters_descending = OrderedDict(sorted(checkin_clusters.items(), key=lambda kv: len(kv[1]), reverse=True))
    cluster_number_center_descending = OrderedDict(sorted(cluster_number_center.items(), key=lambda kv: kv[1][0], reverse=True))
    return checkin_clusters_descending, cluster_number_center_descending

def checkin_distribution_grid(range_coordinates, resolution, checkinData):
    """
    Plot checkin distributions using seaborn package:　http://seaborn.pydata.org/tutorial/distributions.html
    """
    range_lat = np.ceil((range_coordinates[1]-range_coordinates[0])/resolution[0])
#   range_lon = np.ceil((range_coordinates[3]-range_coordinates[2])/resolution[1])
    checkinData.loc[:, 'latGrid'] = checkinData.apply(lambda x: np.floor(x['gridID']/range_lat), axis=1)
    checkinData.loc[:, 'lonGrid'] = checkinData.apply(lambda x: x['gridID']-(x['latGrid']*range_lat), axis=1)
    checkindistributionGrid = checkinData[['latGrid', 'lonGrid']]
    return checkindistributionGrid

def plot_user_locations(uid, df_results, range_coordinates, zoom_map, alg):
    """
    Plot locations using pympas packages
    """

    mymap = pygmaps.maps(np.mean(range_coordinates[:2]), np.mean(range_coordinates[2:]), zoom_map)
    #mymap.setgrids(39.7627, 40.028974, 0.03, 116.182325,116.5787, 0.03)

    # color: e.g. red "#FF0000", Blue "#0000FF", Green "#00FF00"
    if (alg == 'freq') | (alg == 'both'):
        for location in df_results[['lat', 'lon', 'type', 'alg']].values:
            if (location[2] == 'single') and (location[3] == 'freq'):
                mymap.addpoint(location[0], location[1], "#00FF00")
            if (location[2] == 'community') and (location[3] == 'freq'):
                mymap.addpoint(location[0], location[1], "#FF0000")

    if (alg == 'cluster') | (alg == 'both'):
        for location in df_results[['lat', 'lon', 'type', 'alg']].values:
            if (location[2] == 'single') and (location[3] == 'cluster'):
                mymap.addpoint(location[0], location[1], "#00F00")
            if (location[2] == 'community') and (location[3] == 'cluster'):
                mymap.addpoint(location[0], location[1], "#0000FF")
    mymap.draw('./locationMap/infering home location_%s.html' % (str(uid)+'_'+alg))
    return True

class Point(object):
    def __init__(self, latitude, longitude):
        self.lat = latitude
        self.lon = longitude
        self.gridID = 0
        self.gridCenter = list()
    def coordinate_to_gridID(self, range_coordinates, resolution):
        """
        return two new columns in dfc named gridID and gridCenter
        ：range_coordinates:list: [low_lat,high_lat,low_lon,high_lon]
        :resolution: the distance resolution [lat_res,lon_res]
        """
        range_lat = np.ceil((range_coordinates[1]-range_coordinates[0])/resolution[0])
    #    range_lon = np.ceil((range_coordinates[3]-range_coordinates[2])/resolution[1])
        grid_lat_index = np.floor((self.lat-range_coordinates[0])/resolution[0])
        grid_lon_index = np.floor((self.lon-range_coordinates[2])/resolution[1])
        self.gridID = range_lat*grid_lat_index + grid_lon_index
        self.gridCenter = [range_coordinates[0] + resolution[0]*(grid_lat_index+0.5), range_coordinates[2] + resolution[1]*(grid_lon_index+0.5)]
        return True


def generate_point_grid(lat, lon, range_coordinates, resolution):
    """
    """
    point_one = Point(lat, lon)
    point_one.coordinate_to_gridID(range_coordinates, resolution)
    return point_one.gridID

def generate_point_gridCenter(lat, lon, range_coordinates, resolution):
    """
    """
    point_one = Point(lat, lon)
    point_one.coordinate_to_gridID(range_coordinates, resolution)
    return point_one.gridCenter

class GowallaData(object):
    def __init__(self):
        self.dfc = pd.DataFrame(index=[], columns=[])
        self.dfv = pd.DataFrame(index=[], columns=[])
        self.dfe = pd.DataFrame(index=[], columns=[])
        self.dfc_filter = pd.DataFrame(index=[], columns=[])
        self.dfc_active = pd.DataFrame(index=[], columns=[])

    def read_checkin_file(self, filename, items):
        """
        :filename: the file containing the check in infomation
        :items: the columns we need
        """
        self.dfc = pd.read_csv(filename, sep='\t', header=None)
        self.dfc.columns = items
        return True
    def read_spots_file(self, filename, items):
        self.dfv = pd.read_csv(filename, sep='\t')
        self.dfv.columns = items
        coords = self.dfv['loc'].replace('[^0-9. -]+', '', regex=True)
        coords = coords.apply(lambda x: x.split())
        self.dfv['v_lat'] = coords.apply(lambda x: float(x[1]))
        self.dfv['v_lon'] = coords.apply(lambda x: float(x[0]))
        self.dfv = self.dfv.drop('loc', 1)
        return True
    def read_edges_file(self, filename, items):
        self.dfe = pd.read_csv(filename, sep='\t', header=None)
        self.dfe.columns = items
        return True


    def filter_checkin_data(self, areaCoordinates, active_threshold, upper_threshold):
        """
        filter the intial read in dfc data frame, and return self.dfc_active
        ：areaCoordinates: the area we want to filter the checkin data
        :active_threshold: keep the users have number of checkins bigger than threshold
        :upper_threshold:remove the public accounts which have more than the number of check ins
        """
        self.dfc_filter = self.dfc[areaCoordinates]
#        dfc['utc'] = dfc['utc'].astype('datetime64[ms]')
#        dfc['year']     = dfc['utc'].apply(lambda x: x.date().year)
#        dfc['month']    = dfc['utc'].apply(lambda x: x.date().month)
#        dfc['day']      = dfc['utc'].apply(lambda x: x.date().day)
#        dfc['date']     = dfc['utc'].apply(lambda x: x.date())
#        dfc['time']     = dfc['utc'].apply(lambda x: time_in_seconds(x.time()))
#        dfc['isotime']  = dfc['utc'].apply(lambda x: x.isoformat() +'Z')
#        dfc = dfc[dfc['date']<=datetime(2010,10,19).date()]
#        dfc.head()
        self.dfc_active = self.dfc_filter.groupby('uid').filter(lambda x: len(x) > active_threshold and len(x) < upper_threshold)
        return True

    def filter_spots_data(self, areaCoordinates):
        """
        ：dfv:initial data frame
        ：areaCoordinates: the area we want to filter the checkin data
        """
        self.dfv = self.dfv[areaCoordinates]
#        dfv.head()
        return True

class GowallaUser(object):
    """GowallaUser class: define all functions related to one user.
    Attributes:
        uid： use id
        friends:　user's friend
        allCommunities = pd.DataFrame(index=[], columns=[])
        dictNodesId = dict()
        dictNodesIdreverse = dict()
        communityPartition = pd.DataFrame(index=[], columns=[])
        partitionCenters = list()
        communityPartitionNumber = 0
        checkin = pd.DataFrame(index=[], columns=[])
        home_by_checkinCenter = list()
    """
    def __init__(self, oneuserid):
        self.uid = oneuserid
        self.friends = np.array([])
        self.allCommunities = pd.DataFrame(index=[], columns=[])
        self.dictNodesId = dict()
        self.dictNodesIdreverse = dict()
        self.communityPartition = pd.DataFrame(index=[], columns=[])
        self.partitionCenters = list()
        self.communityPartitionNumber = 0
        self.checkin = pd.DataFrame(index=[], columns=[])
        self.home_by_checkinCenter = list()


    def find_all_friends_ID(self, dfe):
        '''
        find all the friends of current user
        '''
        dfe_friends = dfe.loc[dfe['source'] == self.uid]
        self.friends = dfe_friends.target.unique()
        return True

    def find_all_communities(self, dfe):
        '''
        find all the links among the friends of the current user
        '''
        self.allCommunities = dfe.loc[dfe['source'].isin(self.friends)&dfe['target'].isin(self.friends)]
        return True

    def generate_node_ID_dict(self):
        '''
        mapping the nodeID of all friends to a new list starting from 0 (requirement of the infomap algorithm)
        '''
        nodesId = list(self.allCommunities.groupby('source').count().index)
        nodesIdcon = (range(len(nodesId)))
        nodesIdcon = [str(x) for x in nodesIdcon]
        self.dictNodesId = dict(zip(nodesId, nodesIdcon))
        return True


    def get_checkin_data(self, dfc, range_coordinates, resolution):
        """
        Get user's checkin data, and add the gridID and gridCenter for each row
        use the generate_point_grid() and generate_point_gridCenter() functions

        """
        self.checkin = dfc[dfc['uid'] == self.uid]
        self.checkin.loc[:, 'gridID'] = self.checkin.apply(lambda x: generate_point_grid(x['lat'], x['lon'], range_coordinates, resolution), axis=1)
        self.checkin.loc[:, 'gridCenter'] = self.checkin.apply(lambda x: generate_point_gridCenter(x['lat'], x['lon'], range_coordinates, resolution), axis=1)
        return True

    def get_community_checkin_data(self, dfc, range_coordinates, resolution, partition_num):
        """
        Get the checkin data of one of the users communities (partition_num is the community num), and add the gridID and gridCenter for each row

        """
        checkin_community = pd.merge(dfc, self.communityPartition[self.communityPartition['communityID'] == partition_num], left_on='uid', right_on='nodeID')[['uid', 'lat', 'lon', 'utc']]
        if len(checkin_community['uid'].values) != 0:
            checkin_community.loc[:, 'gridID'] = checkin_community.apply(lambda x: generate_point_grid(x['lat'], x['lon'], range_coordinates, resolution), axis=1)
#            communityCheckinDist = user.checkin_distribution_grid(ny_range_coordinates,ny_resolution,checkin_community)
        return checkin_community
    def get_home_location_by_checkin_center(self):
        """
        find the grid which has the most checkin data, and calculate the center of all check in locations in this grid
        """
        home_grid = self.checkin.groupby('gridID').size().idxmax()
        avg_checkin = self.checkin[self.checkin['gridID'] == home_grid].mean()
        self.home_by_checkinCenter = [avg_checkin.lat, avg_checkin.lon]
        return True

    def get_checkin_dist_to_home(self):

        """
        find the grid which has the most checkin data, and calculate the center of all check in locations in this grid
        """
        self.checkin.loc[:, 'disHome'] = self.checkin.apply(lambda x: vincenty([x['lat'], x['lon']], self.home_by_checkinCenter).km, axis=1)
        return True


    def get_home_location_by_community(self, dfc, range_coordinates, resolution):
        """
        Get home location by treating all users in one community as one user
        """
        self.communityPartitionNumber = len(self.communityPartition.communityID.unique())
        self.partitionCenters = dict()
        for partition in range(1, self.communityPartitionNumber+1):

            checkin_community = pd.merge(dfc, self.communityPartition[self.communityPartition['communityID'] == partition], left_on='uid', right_on='nodeID')[['uid', 'lat', 'lon']]
            if len(checkin_community.uid.unique()) > 1:
                ## ==1  means only have check in data of the current user in this community
                checkin_community['gridID'] = checkin_community.apply(lambda x: generate_point_grid(x['lat'], x['lon'], range_coordinates, resolution), axis=1)
                checkin_community['gridCenter'] = checkin_community.apply(lambda x: generate_point_gridCenter(x['lat'], x['lon'], range_coordinates, resolution), axis=1)
                community_home_grid = checkin_community.groupby('gridID').size().idxmax()
                avg_checkin = checkin_community[checkin_community['gridID'] == community_home_grid].mean()
                self.partitionCenters[partition] = [avg_checkin.lat, avg_checkin.lon]
        return True


if __name__ == '__main__':

    # start setting parameters

    ny_range_coordinates = [40.4774, 40.9176, -74.2589, -73.7004]
    ny_resolution = [0.04, 0.04] ## Lat 0.01 --1.11km, lon 0.01 --0.863km
    zoom = 11
    min_checkin_of_active_user = 100
    max_checkin_of_active_user = 2000

    lower_time_bound = 8*60 ## 8 am
    upper_time_bound = 20*60 ## 20pm\
    userid = 0

    # parameters settings ends

    """
    Checkin Data process
    """
    gowalla = GowallaData()
    check_in_file = './datasets/Gowalla_totalCheckins.txt'
    items_checkin = ['uid', 'utc', 'lat', 'lon', 'vid']
    gowalla.read_checkin_file(check_in_file, items_checkin)


    ## filter out users in new york area
    ny = (gowalla.dfc['lat'] >= ny_range_coordinates[0]) & (gowalla.dfc['lat'] <= ny_range_coordinates[1]) & (gowalla.dfc['lon'] >= ny_range_coordinates[2]) & (gowalla.dfc['lon'] <= ny_range_coordinates[3])

    ## filter out the active users
    gowalla.filter_checkin_data(ny, min_checkin_of_active_user, max_checkin_of_active_user)

#    dfc = gowalla.dfc_active
#
#
#    dfc_night = gowalla.filter_night_checkin_data(dfc,lower_time_bound,upper_time_bound)

    user = GowallaUser(gowalla.dfc_active.uid.unique()[userid])

    print user.uid

    user.get_checkin_data(gowalla.dfc_active, ny_range_coordinates, ny_resolution)

    # clustering the night checkin data of a single user.
    dfc_night = filter_night_checkin_data(user.checkin, lower_time_bound, upper_time_bound)
    threshold = 0.002 # about 160m
    metric_cluster = 'euclidean'
    criterion_cluster = 'distance'
    method_link = 'single'
    geopoint_user = np.array(dfc_night[['lat', 'lon']])
    user_clusters = fclusterdata(geopoint_user, threshold, criterion=criterion_cluster, metric=metric_cluster, depth=1, method=method_link)

#    xlim =
#    ylim =
    plt.scatter(geopoint_user[:, 0], geopoint_user[:, 1], c=user_clusters)


    user_checkin_clusters, cluster_num_center_des = checkin_clustering(dfc_night, threshold, criterion_cluster, metric_cluster, method_link)
    first_three = cluster_num_center_des.values()[0:3]

