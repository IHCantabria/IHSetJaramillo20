import numpy as np
import xarray as xr
import pandas as pd
import fast_optimization as fo
from .jaramillo20 import jaramillo20
import json

class Jaramillo20_run(object):
    """
    Jaramillo20_run
    
    Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['Jaramillo20'])

        self.vlt = cfg['vlt']


        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.time = pd.to_datetime(data.time.values)
            self.E = self.hs ** 2
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.time = pd.to_datetime(data.time.values)
            self.E = self.hs ** 2
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]

        if cfg['switch_Yini'] == 1:
            self.Yini = cfg['Yini']
        else:
            ii = np.argmin(np.abs(self.time_obs - self.time[0]))
            self.Yini = self.Obs[ii]
        
        data.close()


        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))

        def run_model(par):
            a = -np.exp(par[0])
            b = np.exp(par[1])
            cacr = -np.exp(par[2])
            cero = -np.exp(par[3])
            vlt = par[4]

            Ymd, _ = jaramillo20(self.E,
                                self.dt,
                                a,
                                b,
                                cacr,
                                cero,
                                self.Yini,
                                vlt)
            return Ymd
        
        self.run_model = run_model
    
    def run(self, par):
        self.full_run = self.run_model(par)
        self.calculate_metrics()

    def calculate_metrics(self):
        self.metrics_names = fo.backtot()[0]
        self.indexes = fo.multi_obj_indexes(self.metrics_names)
        self.metrics = fo.multi_obj_func(self.Obs, self.full_run, self.indexes)



