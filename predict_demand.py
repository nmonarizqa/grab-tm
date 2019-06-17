import pandas as pd
import datetime as dt
import sys
import warnings
from prep_data import augment_data
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor

def predict_demand_geo(base, by_t, the_geo):
    """
    This function predict the demand for a specific geohash6
    """
    rf = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=10,
                            min_samples_leaf=5, min_samples_split=2)
    cond = (base.geohash6==the_geo)
    y = base.demand
    X = base.drop(["demand","geohash6","day","timestamp"], axis=1)
    rf.fit(X[cond],y[cond])
    base_= base[(base.geohash6==the_geo)].copy()

    def create_next_X(base_, by_t):
        """
        Create test data for the next 15-minutes period "t"
        """
        selected_periods = [1,2,3,4,5,6,7,8,96,192]
        base_next  = base_.sort_values("t").tail(1).copy()
        last_timestamp = dt.datetime.strptime(base_next.timestamp.values[0], "%H:%M")
        next_timestamp = last_timestamp+dt.timedelta(minutes=15)
        base_next['day'] = base_next['day']+(next_timestamp-last_timestamp).days
        base_next['t'] = base_next['t']+1
        last_period = base_next['t'].values[0]
        base_next['hour'] = next_timestamp.hour
        base_next['minutes'] = next_timestamp.minute
        base_next['timestamp'] = base_next.hour.apply(str)+":"+base_next.minutes.apply(str)
        base_next['dayofweek'] = (base_next['dayofweek']+(next_timestamp-last_timestamp).days)%7
        for p in selected_periods:
             base_next['d-'+str(p)] = by_t[last_period-p][the_geo]
        #base_next['timestamp'] = 
        result_base = base_next[['geohash6','day','timestamp']]
        X_next = base_next[X.columns]

        return base_next, result_base,last_period
    
    next_5_results = []
    for i in range(5):
        base_next,result_next, last_period = create_next_X(base_, by_t)
        base_ = pd.concat([base_,base_next])
        prediction = rf.predict(base_next[X.columns])
        result_next['demand'] = prediction
        by_t.loc[the_geo,last_period] = prediction
        next_5_results.append(result_next)
    return pd.concat(next_5_results)

if __name__ == "__main__":
  filename = sys.argv[1]
  print "Opening "+filename+"..."
  data = pd.read_csv(filename)

  # prep data
  print "Preparing data..."
  base, by_t = augment_data(data)

  # predict demand for each geohash
  results = []
  print "Predicting demand..."
  n = len(base.geohash6.unique())
  for i,geo in enumerate(base.geohash6.unique()):
    result_geo = predict_demand_geo(base, by_t, geo)
    results.append(result_geo)
    print "\r"+str((i+1)*100./n)[:4]+"%",
  all_results = pd.concat(results)
  print "Success predicting demand."
  all_results.to_csv("prediction.csv", index=False)
  print "Prediction was saved to prediction.csv."



