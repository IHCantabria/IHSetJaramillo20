
# IHSetJaramillo20
Python package to run and calibrate Jaramillo et al. (2020) equilibrium-based shoreline evolution model.

## :house: Local installation
* Using pip:
```bash

pip install git+https://github.com/defreitasL/IHSetJaramillo20.git

```

---
## :zap: Main methods

* [jaramillo20](./IHSetJaramillo20/jaramillo20.py):
```python
# model's it self
jaramillo20(E, dt, a, b, cacr, cero, Yini, vlt)
```
* [cal_Jaramillo20](./IHSetJaramillo20/calibration.py):
```python
# class that prepare the simulation framework
cal_Jaramillo20(path)
```



## :package: Package structures
````

IHSetJaramillo20
|
├── LICENSE
├── README.md
├── build
├── dist
├── IHSetJaramillo20
│   ├── calibration.py
│   └── jaramillo20.py
└── .gitignore

````

---

## :incoming_envelope: Contact us
:snake: For code-development issues contact :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria)

## :copyright: Credits
Developed by :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria).
