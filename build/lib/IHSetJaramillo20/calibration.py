import numpy as np
import xarray as xr
from datetime import datetime
import spotpy as spt
from spotpy.parameter import Uniform
from .jaramillo20 import jaramillo20

class cal_Jaramillo20(object):
    """
    cal_Jaramillo20
    
    Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
    This class reads input datasets, performs calibration, and writes the results to an output NetCDF file.
    
    Note: The function internally uses the Yates09 function for shoreline evolution.
    
    """

    def __init__(self, path):

        self.path = path
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.dt = cfg['dt'].values
        self.switch_Yini = cfg['switch_Yini'].values
        self.switch_vlt = cfg['switch_vlt'].values

        self.n_pop = cfg['n_pop'].values
        self.generations = cfg['generations'].values

        if self.switch_vlt == 0:
            vlt = cfg['vlt'].values

        self.Hs = wav['Hs'].values
        self.Tp = wav['Tp'].values
        self.Dir = wav['Dir'].values
        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)
        self.E = self.Hs ** 2

        self.Y_obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        if self.switch_Yini == 0:
            self.Yini = self.Y_obs[0]

        cfg.close()
        wav.close()
        ens.close()

        
class setup_NSGAII(object):
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
            
    def parameter(self):
        return spt.parameter.generate(self.params)

    def simulation(self, par):
        
        a = np.exp(par[0][0])
        b = np.exp(par[1][0])
        cacr = np.exp(par[2][0])
        cero = np.exp(par[3][0])

        Ymd = jaramillo20(self.ja20_obj.E,
                          self.ja20_obj.dt,
                          a,
                          b,
                          cacr,
                          cero,
                          self.ja20_obj.Yini,
                          self.ja20_obj.vlt)
        
        return Ymd[self.ja20_obj.idx_obs]
    
    def evaluation(self):
        return self.ja20_obj.Y_obs
            
    def objectivefunction(self, simulation, evaluation):
        return self.obj_unc(simulation, evaluation)
    
    def setUp(self):
        
        self.sampler = spt.algorithms.NSGAII(
                    spot_setup=self, dbname="NSGA2"
        )
        self.sampler.sample(
                            self.ja20_obj.generations,
                            n_obj=3,
                            n_pop=self.ja20_obj.n_pop
                            )

        results = self.sampler.getdata()
        return results

