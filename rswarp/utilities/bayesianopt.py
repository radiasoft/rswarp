import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.optimize import minimize, basinhopping
from scipy.special import erf


class BayesianOptimizer:
    # should be (samples, parameters)
    def __init__(self, func, x0, y0, ranges):
        self._X = x0
        self._y = y0
        self._f_star = np.max(y0)
        self._f_star_index = np.argmax(y0)

        self.range = ranges
        self._eval = func

        self.kernel = Matern(length_scale=1., nu=1.5)
        self.gpr = gpr(kernel=self.kernel, n_restarts_optimizer=10)
        self.gpr.fit(x0, y0)

        self.xi = 3.0

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, x):
        self._X = np.append(self._X, x).reshape(-1, 1)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = np.append(self._y, y).reshape(-1, 1)
        self._f_star_index = np.argmax(self.y)
        self._f_star = self.y[self._f_star_index]

    def _return_ei(self, x):
        x = np.array(x).reshape(-1, 1)

        mean, std = self.gpr.predict(x, return_std=True)
        pdf = np.exp(-(mean - self._f_star - self.xi) ** 2 / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)
        cdf = erf((mean - self._f_star - self.xi) / std)

        EI = mean - self._f_star * cdf + std * pdf

        return -EI

    def evaluate(self, x):
        x = x.reshape(-1, 1)
        y = self._eval(x)
        self.X = x
        self.y = y
        self.gpr.fit(self.X, self.y)

    def optimize(self, n=1):
        for _ in range(n):
            print('fstar', self._f_star)
            #             res = minimize(self._return_ei, x0=(self.X[self._f_star_index],),
            #                            method='SLSQP', bounds=(self.range,))
            mybounds = MyBounds()
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": (self.range,)}
            res = basinhopping(self._return_ei, x0=(self.X[self._f_star_index],), T=5.0, stepsize=4,
                               accept_test=mybounds, minimizer_kwargs=minimizer_kwargs)
            print('optimizer result, new x:', res['x'])
            if np.any(np.abs(self.X - res['x']) < np.abs(self.range[1] - self.range[0]) / 1e4):
                print('RANDOM', np.abs(self.X - res['x']), np.abs(self.range[1] - self.range[0]) / 1.0e4)
                result = np.array(np.random.uniform(low=self.range[0], high=self.range[1]))
            else:
                result = res['x']
            self.evaluate(result)

    def get_model(self):
        vals = np.linspace(*self.range, 1000).reshape(-1, 1)

        return vals, self.gpr.predict(vals, return_std=True)

class MyBounds(object):
    def __init__(self, xmin, xmax):
        """

        Args:
            xmin: (list) minimum values
            xmax: (list) maximum values
        """
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin