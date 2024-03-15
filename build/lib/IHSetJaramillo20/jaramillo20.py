import numpy as np
import xarray as xr
from datetime import datetime
from numba import jit
import spotpy as spt
from spotpy.parameter import Uniform
from IHSetCalibration import *
@jit
def jaramillo20(E, dt, a, b, cacr, cero, Yini, vlt):
    """
    Jaramillo et al. 2020 model
    """
    Seq = (E -b)/a
    Y = np.zeros_like(E)
    Y[0] = Yini
    for i in range(1, len(E)):
        if Y[i-1] < Seq[i]:
            Y[i] = ((Y[i-1]-Seq[i])*np.exp(-1 * a *cacr *(E[i] ^ 0.5)*dt))+Seq[i] + vlt*dt
        else:
            Y[i] = ((Y[i-1]-Seq[i])*np.exp(-1 * a *cero *(E[i] ^ 0.5)*dt))+Seq[i] + vlt*dt

    return Y



# class cal_Jaramillo20(object):
#     """
#     cal_Jaramillo20
    
#     Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
#     This class reads input datasets, performs calibration, and writes the results to an output NetCDF file.
    
#     Note: The function internally uses the Yates09 function for shoreline evolution.
    
#     """

#     def __init__(self, path):

#         self.path = path
        
#         mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

#         cfg = xr.open_dataset(path+'config.nc')
#         wav = xr.open_dataset(path+'wav.nc')
#         ens = xr.open_dataset(path+'ens.nc')

#         self.cal_alg = cfg['cal_alg'].values
#         self.dt = cfg['dt'].values
#         self.switch_Yini = cfg['switch_Yini'].values
#         self.switch_vlt = cfg['switch_vlt'].values

#         if self.switch_vlt == 0:
#             vlt = cfg['vlt'].values

#         self.Hs = wav['Hs'].values
#         self.Tp = wav['Tp'].values
#         self.Dir = wav['Dir'].values
#         self.time = mkTime(wav['Year'].values, wav['Month'].values, wav['Day'].values, wav['Hour'].values)
#         self.E = Hs ^ 2

#         self.Y_obs = ens['Yobs'].values
#         self.time_obs = mkTime(ens['Year'].values, ens['Month'].values, ens['Day'].values, ens['Hour'].values)

#         if self.switch_Yini == 0:
#             self.Yini = self.Y_obs[0]

#         cfg.close()
#         wav.close()
#         ens.close()

#     def calibrate(self):
#         if self.cal_alg == 'NSGAII':
#             cfg = xr.open_dataset(self.path+'config.nc')
#             # Number of generations
#             generations = cfg['generations'].values
#             # Number of individuals in the population
#             n_pop = cfg['n_pop'].values
#             from IHSetCalibration import multi_obj_func
#             def obj(params):
#                 Ymd = jaramillo20(self.E,
#                                   self.dt,
#                                   params[0],
#                                   params[2],
#                                   params[3],
#                                   params[4],
#                                   self.Yini,
#                                   self.vlt)
#                 YYsl = Ymd[self.time == self.time_obs]
#                 return multi_obj_func(YYsl, self.Y_obs)
            
#             sp_setup = spt.spot_setup(obj_func=obj)
#             sampler = spt.algorithms.NSGAII(
#                 spot_setup=sp_setup, dbname="NSGA2"
#             )
#             sampler.sample(generations, n_obj=3, n_pop=n_pop)

#         results = sampler.getdata()
        
class spt_setup_NSGAII(object):
    """
    spt_setup
    
    Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
    This class reads input datasets, performs calibration, and writes the results to an output NetCDF file.
    
    Note: The function internally uses the Yates09 function for shoreline evolution.
    
    """

    def __init__(self, multi_obj_func, ja20_obj):
        self.obj_func = multi_obj_func
        self.ja20_obj = ja20_obj
        if ja20_obj.switch_vlt == 0:
            self.params = [
                Uniform('a', np.log(1e-3), np.log(5e-1)),
                Uniform('b', np.log(1e-1), np.log(1e+2)),
                Uniform('cacr', np.log(1e-5), np.log(1e-1)),
                Uniform('cero', np.log(1e-5), np.log(1e-1))
            ]
        # if self.cal_alg == 'NSGAII':
        # cfg = xr.open_dataset(self.path+'config.nc')
        # Number of generations
        # generations = cfg['generations'].values
        # Number of individuals in the population
        # n_pop = cfg['n_pop'].values

    def simulation(self, params):
        

        Ymd = jaramillo20(self.E,
                          self.dt,
                          params[0],
                          params[2],
                          params[3],
                          params[4],
                          self.Yini,
                          self.vlt)
        
        return Ymd[self.ja20_obj.idx_obs]
    
    def evaluation(self):
        return self.ja20_obj.Y_obs
            
    def objectivefunction(self, simulation, evaluation):
        return self.obj_unc(simulation, evaluation)
    
    def setUp(self):
        self.sp_setup = spt.spot_setup(
                    obj_func=self.objectivefunction,
                    simulation=self.simulation,
                    evaluation=self.evaluation,
                    params=self.params
        )
        self.sampler = spt.algorithms.NSGAII(
                    spot_setup=self.sp_setup, dbname="NSGA2"
        )
        self.sampler.sample(self.generations, n_obj=3, n_pop=self.n_pop)

        results = sampler.getdata()
        return results