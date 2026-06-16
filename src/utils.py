import numpy as np

def bootstrap_ci(
    data, 
    statfunction=np.average, 
    alpha = 0.05, 
    n_samples = 500):
    """inspired by https://github.com/cgevans/scikits-bootstrap"""
    import warnings
    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0],))    
    alphas = np.array([alpha/2, 1 - alpha/2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)
    if np.any(nvals < 10) or np.any(nvals >= n_samples-10):
        warnings.warn("Small number of bootstrap samples for the requested alpha. "
                      "Try to increase n_samples")
    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()
    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    stat.sort(axis=0)
    return stat[nvals]

def bootstrap_uci(data):
    if len(data) == 0: return np.nan
    return bootstrap_ci(data)[1]

def bootstrap_lci(data):
    if len(data) == 0: return np.nan
    return bootstrap_ci(data)[0]

def load_observed_data(filepath="data/air_monitor_2022_03_30.csv"):
    """Load and clean the real-world PM2.5 data"""
    import pandas as pd
    import datetime
    
    dt = pd.read_csv(filepath)
    dt.columns = ['date_time', 'PM10', "PM2p5", "CO2", "Temperature", "Humidity"]
    dt['datetime'] = pd.to_datetime(dt.date_time, format='%d/%m/%Y %H:%M')
    
    startdate = datetime.datetime(2022, 1, 15)
    enddate = datetime.datetime(2022, 3, 25, 14, 30)
    
    dt = dt[(dt.datetime > startdate) & (dt.datetime < enddate)]
    dt['datetime_of_experiment'] = dt['datetime'] - startdate
    
    # Logic: 2-day cycles (1 day Off/A, 1 day On/B)
    dt['purifier_on'] = dt.datetime_of_experiment.dt.days % 2 == 1
    dt['period'] = np.floor(dt.datetime_of_experiment.dt.days / 2)
    
    # Mirror the censoring logic (exclude pre-3am data)
    dt['PM2p5_censored'] = dt.PM2p5
    dt.loc[dt.datetime.dt.time <= datetime.time(3,00), 'PM2p5_censored'] = np.nan
    
    # Map True/False to A/B to match simulation
    # Simulation logic: 
    #   Phase A = Baseline (Purifier OFF)
    #   Phase B = Treatment (Purifier ON)
    dt['phase'] = dt['purifier_on'].map({True: 'B', False: 'A'})
    # Load resistance training dates
    rt_dates = pd.read_csv('data/resistance_training_dates.csv')['date']
    rt_dates = pd.to_datetime(rt_dates, format='%d/%m/%Y').dt.date
    dt['resistance_training'] = dt['datetime'].dt.date.isin(rt_dates).map({True: 'Yes', False: 'No'})
    
    return dt.dropna(subset=['PM2p5_censored'])
