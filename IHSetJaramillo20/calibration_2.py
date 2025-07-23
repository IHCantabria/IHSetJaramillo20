import numpy as np
from .jaramillo20 import jaramillo20
from IHSetUtils.CoastlineModel import CoastlineModel

class cal_Jaramillo20_2(CoastlineModel):
    """
    cal_Jaramillo20_2
    
    Configuration to calibrate and run the Jaramillo et al. (2020) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Jaramillo et al. (2020)',
            mode='calibration',
            model_type='CS',
            model_key='Jaramillo20'
        )

        self.setup_forcing()

    def setup_forcing(self):

        self.switch_Yini = self.cfg['switch_Yini']
        self.switch_vlt = self.cfg['switch_vlt']
        if self.switch_vlt == 0:
            self.vlt = self.cfg['vlt']
            
        self.E = self.hs ** 2
        self.E_s = self.hs_s ** 2

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

    def init_par(self, population_size: int):
        if self.switch_Yini == 0 and self.switch_vlt == 0:
            lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
            uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])
        elif self.switch_Yini == 1 and self.switch_vlt == 0:
            lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3]), 0.75 * np.min(self.Obs_splited)])
            uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3]), 1.25 * np.max(self.Obs_splited)])
        elif self.switch_Yini == 0 and self.switch_vlt == 1:
            lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3]), self.lb[4]])
            uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3]), self.ub[4]])
        elif self.switch_Yini == 1 and self.switch_vlt == 1:
            lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3]), self.lb[4], 0.75 * np.min(self.Obs_splited)])
            uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3]), self.ub[4], 1.25 * np.max(self.Obs_splited)])
        pop = np.zeros((population_size, len(lowers)))        
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers


    def model_sim(self, par: np.ndarray) -> np.ndarray:
        a = -np.exp(par[0])
        b = par[1]
        cacr = -np.exp(par[2])
        cero = -np.exp(par[3])
        
        if self.switch_Yini == 0 and self.switch_vlt == 0:
            vlt = self.vlt
            Yini = self.Yini
        elif self.switch_Yini == 1 and self.switch_vlt == 0:
            vlt = self.vlt
            Yini = par[4]
        elif self.switch_Yini == 0 and self.switch_vlt == 1:
            vlt = par[4]
            Yini = self.Yini
        elif self.switch_Yini == 1 and self.switch_vlt == 1:
            vlt = par[4]
            Yini = par[5]
        Ymd, _ = jaramillo20(self.E_s,
                                self.dt_s,
                                a,
                                b,
                                cacr,
                                cero,
                                Yini,
                                vlt)
        return Ymd[self.idx_obs_splited]
    
    def run_model(self, par: np.ndarray) -> np.ndarray:
        a = par[0]
        b = par[1]
        cacr = par[2]
        cero = par[3]
        if self.switch_Yini == 0 and self.switch_vlt == 0:
            vlt = self.vlt
            Yini = self.Yini
        elif self.switch_Yini == 1 and self.switch_vlt == 0:
            vlt = self.vlt
            Yini = par[4]
        elif self.switch_Yini == 0 and self.switch_vlt == 1:
            vlt = par[4]
            Yini = self.Yini
        elif self.switch_Yini == 1 and self.switch_vlt == 1:
            vlt = par[4]
            Yini = par[5]
        Ymd, _ = jaramillo20(self.E,
                            self.dt,
                            a,
                            b,
                            cacr,
                            cero,
                            Yini,
                            vlt)
        return Ymd

    def _set_parameter_names(self):
        if self.switch_Yini == 0 and self.switch_vlt == 0:
            self.par_names = [r'a', r'b', r'C+', r'C-']
        elif self.switch_Yini == 1 and self.switch_vlt == 0:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'Y_i']
        elif self.switch_Yini == 0 and self.switch_vlt == 1:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'v_lt']
        elif self.switch_Yini == 1 and self.switch_vlt == 1:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'v_lt', r'Y_i']
        for idx in [0, 2, 3]:
            self.par_values[idx] = -np.exp(self.par_values[idx])
