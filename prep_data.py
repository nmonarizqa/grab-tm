import pandas as pd
import numpy as np
import geohash
from sklearn.cluster import KMeans
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

def get_14_days_data(data):
    """
    This function returns data from the past 14 days
    """
    dem = data.sort_values(["day","timestamp"])
    dem.index = range(len(dem))
    last_entry = dem.iloc[-1]
    last_day = last_entry.day
    last_time = dt.datetime.strptime(last_entry.timestamp,"%H:%M")
    next_time = last_time+dt.timedelta(minutes=15)
    next_time_str = str(next_time.hour)+":"+str(next_time.minute)
    last_14_day_time = last_time-dt.timedelta(days=14)+dt.timedelta(minutes=15)
    last_14_day_time_str = str(last_14_day_time.hour)+":"+str(last_14_day_time.minute)
    next_day = last_day+next_time.day-1
    last_14_day = next_day-14
    try:
        start_index = dem[(dem.day==last_14_day)&(dem.timestamp==last_14_day_time_str)].index[0]
        train = dem.loc[start_index:]
    except:
        train = dem
        
    def prepare_data(data):
        data['hour'] = data.timestamp.apply(lambda x: int(x.split(":")[0]))
        data['minutes'] = data.timestamp.apply(lambda x: int(x.split(":")[1]))
        data['t'] = ((data['day']-1)*(24*60)+data['hour']*60+data['minutes'])/15
        data['t_day']  = (data['t']%(24*4))
        data['t_week']  = (data['t']%(24*28))
        data['dayofweek'] = data.day%7
        data = data.sort_values(["geohash6","t"])
        data.index = range(len(data))
        return data
    train = prepare_data(train)
    last_t = train[train.timestamp==last_entry.timestamp].t.max()
    train = train[train['t']<=last_t]

    return train, next_day, next_time_str

def prepare_data_for_cluster(data):
    """
    This function return normalized data for each day, each geohash6
    """
    by_t_day = data.groupby(["geohash6","dayofweek","t_day"]).demand.mean().unstack().fillna(0)
    mean_across_t = by_t_day.mean()
    std_across_t = by_t_day.std()
    by_t_day_norm = (by_t_day-mean_across_t)/std_across_t
    mean_across_geo = by_t_day_norm.mean(axis=1)
    std_across_geo = by_t_day_norm.std(axis=1)
    by_t_day_norm = (by_t_day_norm.T-mean_across_geo)/std_across_geo
    return by_t_day_norm

def cluster(data_norm):
    """
    For each geohash, each day, we learn the daily pattern series
    and then put them into a separate cluster --> label them based on the cluster
    """
    train = data_norm.T
    km = KMeans(random_state=1, n_clusters=8)
    clusters = km.fit_predict(train)
    clustered = pd.Series(clusters, index=train.index)
    clustered_piv = clustered.unstack().T
    clustered_piv = clustered_piv.fillna(clustered_piv.median(axis=0))
    result = clustered_piv.unstack().reset_index().rename(columns={0:'time_cluster'})
    return result

def get_fourier(data_piv):
    """
    For each geohash, we get the top 5 fourier values
    """
    freq = np.fft.rfftfreq(len(data_piv.columns), 1.0)
    ffs = []
    top_5_freqs_dict = {}
    top_5_amplis_dict = {}
    for i in range(len(data_piv)):
        the_data = data_piv.iloc[i]
        f = np.abs(np.fft.rfft(the_data))
        top_5 = pd.Series(f, index=freq).iloc[1:].sort_values(ascending=False).head(5)
        top_5_freqs_dict[the_data.name] = top_5.index
        top_5_amplis_dict[the_data.name] = top_5.values
        ffs.append(f)
        
    ffs_df_ = pd.DataFrame(ffs,index=data_piv.index,columns=freq)
    agg_fouriers_df = ffs_df_.iloc[:, 1:].max().sort_values(ascending=False).reset_index()
    top_5_freqs_df = pd.DataFrame(top_5_freqs_dict, index=["fft_f_"+str(i) for i in range(5)]).T
    top_5_amplis_df = pd.DataFrame(top_5_amplis_dict, index=["fft_a_"+str(i) for i in range(5)]).T
    return top_5_freqs_df, top_5_amplis_df, agg_fouriers_df

def get_n_previous_demand(df, list_n):
    """
    Get the previous demand values (t-1, t-2, etc) based on list_n
    """
    demand_shifts = []
    for i in list_n:
        the_shift = df.T.shift(i).unstack()
        the_shift.name = "d-"+str(i)
        demand_shifts.append(the_shift)
    demand_shifts_df = pd.concat(demand_shifts, axis=1).dropna()
    return demand_shifts_df

def augment_data(raw_data):
  """
  raw_data is whatever data provided (input) from the csv file
  This function converts raw_data into train data
  """
  temp, n_day, n_time = get_14_days_data(raw_data)
  temp_norm = prepare_data_for_cluster(temp)
  clustered = cluster(temp_norm)

  # get demand for eact unique time "t"
  by_t = temp.pivot_table(index="geohash6",columns="t",values="demand").fillna(0)

  # get fourier values
  top_5_freqs, top_5_amplis, agg_fouriers = get_fourier(by_t)

  # get latlon for each geohash
  latlon = map(lambda x: geohash.decode_exactly(x), top_5_amplis.index)
  loc = pd.DataFrame({
    'lat': [x[1] for x in latlon],
    'lon': [x[0] for x in latlon]
    },
    index=top_5_amplis.index)
  
  # merge fourier and location
  var_add = pd.concat([top_5_freqs, top_5_amplis, loc], axis=1)

  # get previous demand values
  selected_periods = [1,2,3,4,5,6,7,8,96,192]
  ds = get_n_previous_demand(by_t, selected_periods)

  # create training data
  base = temp[["geohash6","t","demand","dayofweek","hour","minutes","timestamp","day"]]
  base = pd.merge(ds.reset_index(), base, how="right", on=["geohash6","t"]).dropna()
  base = pd.merge(base, clustered, on=["geohash6","dayofweek"], how="left")
  base = pd.merge(base, var_add.reset_index().rename(columns={'index':'geohash6'}), how="left", on="geohash6")
  base = base.sort_values(["geohash6","t"])
  base.index = range(len(base))

  # return training data and table of all previous demands
  return base, by_t


