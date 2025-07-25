# import numpy as np
# import xarray as xr
# import pandas as pd
# import fast_optimization as fo
# from .jaramillo20 import jaramillo20_njit
# import json

# class Jaramillo20_run(object):
#     """
#     Jaramillo20_run
    
#     Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
#     This class reads input datasets, performs its calibration.
#     """

#     def __init__(self, path):

#         self.path = path
#         self.name = 'Jaramillo et al. (2020)'
#         self.mode = 'standalone'
#         self.type = 'CS'
     
#         data = xr.open_dataset(path)
        
#         cfg = json.loads(data.attrs['run_Jaramillo20'])

#         self.switch_Yini = cfg['switch_Yini']

#         self.cfg = cfg

#         if cfg['trs'] == 'Average':
#             self.hs = np.mean(data.hs.values, axis=1)
#             self.time = pd.to_datetime(data.time.values)
#             self.E = self.hs ** 2
#             self.Obs = data.average_obs.values
#             self.Obs = self.Obs[~data.mask_nan_average_obs]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_average_obs]
#         else:
#             self.hs = data.hs.values[:, cfg['trs']]
#             self.time = pd.to_datetime(data.time.values)
#             self.E = self.hs ** 2
#             self.Obs = data.obs.values[:, cfg['trs']]
#             self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]

#         self.start_date = pd.to_datetime(cfg['start_date'])
#         self.end_date = pd.to_datetime(cfg['end_date'])

#         data.close()

#         self.split_data()

#         if self.switch_Yini == 1:
#             ii = np.argmin(np.abs(self.time_obs - self.time[0]))
#             self.Yini = self.Obs[ii]
        
#         mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        
#         self.idx_obs = mkIdx(self.time_obs)

#         # Now we calculate the dt from the time variable
#         mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
#         self.dt = mkDT(np.arange(0, len(self.time)-1))

#         if self.switch_Yini == 0:
#             def run_model(par):
#                 a = par[0]
#                 b = par[1]
#                 cacr = par[2]
#                 cero = par[3]
#                 vlt = par[4]
#                 Yini = par[5]

#                 Ymd, _ = jaramillo20_njit(self.E,
#                                     self.dt,
#                                     a,
#                                     b,
#                                     cacr,
#                                     cero,
#                                     Yini,
#                                     vlt)
#                 return Ymd
        
#             self.run_model = run_model
#         else:
#             def run_model(par):
#                 a = par[0]
#                 b = par[1]
#                 cacr = par[2]
#                 cero = par[3]
#                 vlt = par[4]

#                 Ymd, _ = jaramillo20_njit(self.E,
#                                     self.dt,
#                                     a,
#                                     b,
#                                     cacr,
#                                     cero,
#                                     self.Yini,
#                                     vlt)
#                 return Ymd
        
#             self.run_model = run_model
    
#     def run(self, par):
#         self.full_run = self.run_model(par)
#         if self.switch_Yini == 1:
#             self.par_names = [r'a', r'b', r'C+', r'C-', r'v_{lt}']
#             self.par_values = par
#         elif self.switch_Yini == 0:
#             self.par_names = [r'a', r'b', r'C+', r'C-', r'v_{lt}', r'Y_{i}']
#             self.par_values = par
#         # self.calculate_metrics()

#     def calculate_metrics(self):
#         self.metrics_names = fo.backtot()[0]
#         self.indexes = fo.multi_obj_indexes(self.metrics_names)
#         self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)

#     def split_data(self):
#         """
#         Split the data into calibration and validation datasets.
#         """
#         ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
#         self.E = self.E[ii]
#         self.time = self.time[ii]

#         ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
#         self.Obs = self.Obs[ii]
#         self.time_obs = self.time_obs[ii]

import numpy as np
from .jaramillo20 import jaramillo20_njit
from IHSetUtils.CoastlineModel import CoastlineModel



class Jaramillo20_run(CoastlineModel):
    """
    Jaramillo20_run
    
    Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Jaramillo et al. (2020)',
            mode='standalone',
            model_type='CS',
            model_key='run_Jaramillo20'
        )

        self.setup_forcing()

    def setup_forcing(self):

        self.switch_Yini = self.cfg['switch_Yini']            
        self.E = self.hs ** 2

        if self.switch_Yini == 1:
            self.Yini = self.Obs[0]
    
    def run_model(self, par: np.ndarray) -> np.ndarray:
        a = par[0]
        b = par[1]
        cacr = par[2]
        cero = par[3]

        if self.switch_Yini == 1:
            vlt = par[4]
            Yini = self.Yini
        elif self.switch_Yini == 0:
            vlt = par[4]
            Yini = par[5]
        Ymd, _ = jaramillo20_njit(self.E,
                            self.dt,
                            a,
                            b,
                            cacr,
                            cero,
                            Yini,
                            vlt)
        return Ymd

    def _set_parameter_names(self):
        if self.switch_Yini == 1:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'v_lt']
        elif self.switch_Yini == 0:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'v_lt', r'Y_i']




