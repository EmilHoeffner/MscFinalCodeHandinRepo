import statsmodels.api as sm
import statsmodels.stats.api as sms

from statsmodels.compat import lzip
import matplotlib.pyplot as plt
import pylab as py

import numpy as np

import math

# https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/

# https://www.stata.com/support/faqs/statistics/logit-transformation/

class LinearRegressor:

    def __init__(self, functional_form):
        self.model = None
        self.functional_form = functional_form

        assert(functional_form == "level" or functional_form == "logit")

    '''
        X: Numpy Array of shape [N, D]
        y: Numpy Array of shape [N,]
        # https://stackoverflow.com/questions/58607044/error-using-statsmodels-api-using-ols-fit
    '''
    # https://stackoverflow.com/questions/46861158/robust-linear-regression-results-in-python-and-stata-do-not-agree
    def fit(self, X, y):
        X = sm.add_constant(X)
        if self.functional_form == "logit":
            for i in range(len(y)):
                if y[i] == 1.0:
                    y[i] -= 0.05
                elif y[i] == 0.0:
                    y[i] += 0.05

                val = y[i] / (1.0 - y[i])
                y[i] = math.log(val)

        self.model = sm.OLS(y, X).fit(cov_type='HC1')

    def summary(self):
        return self.model.summary()
    
    # https://www.statsmodels.org/v0.10.2/examples/notebooks/generated/predict.html
    def predict(self, X):
        X = sm.add_constant(X)
        ypred = self.model.predict(X)
        return ypred
    
    # https://www.statology.org/breusch-pagan-test-python/
    '''
        Returns p-value of the test
    '''
    def is_heteroskedastic(self):
        names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
        test = sms.het_breuschpagan(self.model.resid, self.model.model.exog)

        print(lzip(names, test))

    # https://www.geeksforgeeks.org/python/how-to-plot-a-normal-distribution-with-matplotlib-in-python/
    # https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/
    # https://www.statsmodels.org/devel/generated/statsmodels.graphics.gofplots.qqplot.html
    def normal_residuals_plot(self):
        residuals = self.model.resid

        plt.hist(x = residuals, bins = 100, density = True)
        plt.show()
        plt.close()

        sm.qqplot(residuals, line = "45")
        plt.show()

    # https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/
    # https://www.geeksforgeeks.org/python/plot-a-horizontal-line-in-matplotlib/
    def fitted_vs_residuals_plot(self):
        fitted_values = self.model.fittedvalues 
        residuals = self.model.resid

        plt.scatter(fitted_values, residuals)
        plt.hlines(y = [0], xmin = 0, xmax = 1, colors = ["r"])
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.show()



