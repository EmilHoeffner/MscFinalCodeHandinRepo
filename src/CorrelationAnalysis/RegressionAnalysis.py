
import os 
import sys 
from dataloader import DataLoader
import numpy as np
from sklearn.metrics import r2_score
from LinearRegression import LinearRegressor
import matplotlib.pyplot as plt


sys.path.append("..") 


# https://www.statsmodels.org/stable/regression.html
# https://www.geeksforgeeks.org/python/linear-regression-in-python-using-statsmodels/
# https://www.lexjansen.com/mwsug/2013/FS/MWSUG-2013-FS05.pdf (consider logit transformation with OLS).
def linear_regression_analysis_summary(features, y, functional_form):
    D = len(features)
    N = len(features[0])

    X = np.zeros((N, D))

    for i, feature in enumerate(features):
        X[:, i] = feature     

    y = np.array(y)

    model = LinearRegressor(functional_form)
    model.fit(X, y)
    print(model.summary())

    return model


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one arguments")
    
    data_set_iden = sys.argv[1]

    dl = DataLoader(data_set_iden)
    CL_train = np.array(dl.CL_train())
    QL_train = np.array(dl.QL_train())
    Sim_train = np.array(dl.load_train_feature("Sim_Max"))
    PA_train = np.array(dl.PA_train())
    y_train = np.array([(f1 + bert) / 2.0 for f1,bert in zip(dl.load_train_feature("F1"), dl.load_train_feature("BERT"))])

    CL_test = np.array(dl.CL_test())
    QL_test = np.array(dl.QL_test())
    Sim_test = np.array(dl.load_test_feature("Sim_Max"))
    PA_test = np.array(dl.PA_test())
    y_test = np.array([(f1 + bert) / 2.0 for f1,bert in zip(dl.load_test_feature("F1"), dl.load_test_feature("BERT"))])
    

    '''
        Linear Regression for Summary Statistics level response variable
    '''
    X_train_summary = [CL_train, QL_train, Sim_train, PA_train]
    linreg_summary_level = linear_regression_analysis_summary(X_train_summary, y_train, "level") 

    X_train_summary = np.vstack((CL_train, QL_train, Sim_train, PA_train)).T
    y_pred_train_summary = linreg_summary_level.predict(X_train_summary)

    r2_train = r2_score(y_true = y_train, y_pred = y_pred_train_summary)
    r2_capped_train = r2_score(y_true = y_train, y_pred = np.clip(y_pred_train_summary, a_min = 0.0, a_max = 1.0))

    print("R2_train = {}".format(r2_train))
    print("R2_capped_train = {}".format(r2_capped_train))
    
    linreg_summary_level.is_heteroskedastic()
    #linreg_summary_level.normal_residuals_plot()
    #linreg_summary_level.fitted_vs_residuals_plot()

    X_test_summary = np.vstack((CL_test, QL_test, Sim_test, PA_test)).T
    y_pred_test_summary = linreg_summary_level.predict(X_test_summary)

    r2_test = r2_score(y_true = y_test, y_pred = y_pred_test_summary)
    r2_capped_test = r2_score(y_true = y_test, y_pred = np.clip(y_pred_test_summary, a_min = 0.0, a_max = 1.0))

    print("\n")

    print("R2_test = {}".format(r2_test))
    print("R2_capped_test = {}".format(r2_capped_test))


    plt.title(f"{data_set_iden}: Predictions vs Labels", fontsize = 18)
    plt.scatter(x = y_train, y = y_pred_train_summary, label = "Train", color = "blue")
    plt.scatter(x = y_test, y = y_pred_test_summary, label = "Test", color = "red")
    plt.xlabel("Ground Truth", fontsize = 18)
    plt.ylabel("Predictions", fontsize = 18)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    


if __name__ == "__main__":
    main()