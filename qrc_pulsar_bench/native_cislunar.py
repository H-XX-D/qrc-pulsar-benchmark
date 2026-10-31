"""Explicit ctypes bridge to the host-compiled cislunar C++ force core.

This executes native CPU code, not FPGA hardware. Compile separately, e.g.
clang++ -O2 -std=c++17 -shared -fPIC hardware/cislunar_force_core.cpp -o core.so
"""
import ctypes
import numpy as np
from .cislunar_forces import CislunarForceParameters


class NativeCislunarKernel:
    def __init__(self, library_path, parameters=CislunarForceParameters()):
        self.library = ctypes.CDLL(str(library_path))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        self.library.cislunar_acceleration.argtypes = [vector] * 4
        self.library.cislunar_acceleration.restype = ctypes.c_int
        self.library.cislunar_verlet_step.argtypes = [vector, vector, ctypes.c_double, vector, vector, vector]
        self.library.cislunar_verlet_step.restype = ctypes.c_int
        self.parameters = parameters.native_parameters()

    @staticmethod
    def _array(value, length):
        x = np.array(value, dtype=np.float64, order="C", copy=True)
        if x.shape != (length,) or not np.all(np.isfinite(x)):
            raise ValueError(f"Expected finite array of length {length}")
        return x

    def acceleration(self, position, environment):
        r = self._array(position, 3)
        env = self._array(np.asarray(environment).reshape(-1), 9)
        out = np.empty(3)
        if self.library.cislunar_acceleration(r, env, self.parameters, out):
            raise ValueError("Native force rejected invalid input or collision")
        return out

    def step(self, position, velocity, dt_s, environment_start, environment_end):
        r, v = self._array(position, 3), self._array(velocity, 3)
        env0 = self._array(np.asarray(environment_start).reshape(-1), 9)
        env1 = self._array(np.asarray(environment_end).reshape(-1), 9)
        if self.library.cislunar_verlet_step(r, v, dt_s, env0, env1, self.parameters):
            raise ValueError("Native step rejected invalid input or collision")
        return r, v
