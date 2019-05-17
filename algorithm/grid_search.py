from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ConstantKernel)
from sklearn.model_selection import GridSearchCV
from algorithms.Tools import *

data = Dataset()  # increased_ys=True

# X, Y = data.get_estimated_data()
X, Y = data.get_no_estimated_data()

test_size = 0.25
trainX, testX, trainY, testY = split_train_test(X, Y, test_size, scale_data=True)

all_results = []
all_predictions = []


def optimize_model(clf, param_grid, name):
    start_time = time()
    print('#####' + name + '#####')
    model = GridSearchCV(clf, cv=5, param_grid=param_grid)
    model.fit(trainX, trainY)
    predY = model.predict(testX).tolist()
    predY = np.resize(predY, (len(testY), 1))
    print('Best parameters: ' + str(model.best_params_))
    print('Best score: ' + str(round(model.best_score_ * 100, 2)) + '%')
    print()

    r2 = r2_score(testY, predY) * 100
    accuracy = model.score(testX, testY) * 100
    MSE = mean_squared_error(testY, predY)
    RMSE = np.sqrt(mean_squared_error(testY, predY))
    MAE = mean_absolute_error(testY, predY)
    MSLE = mean_squared_log_error(abs(testY), abs(predY))
    MEDAE = median_absolute_error(testY, predY)
    MAPE = mean_absolute_percentage_error(testY, predY)
    EVS = explained_variance_score(testY, predY) * 100
    RMSE_norm = root_mean_squared_error_norm(testY, predY) * 100
    MEDAE_norm = median_absolute_error_norm(testY, predY) * 100
    MSLE_norm = mean_squared_log_error_norm(testY, predY) * 100

    Mean_pred = np.mean(predY)
    Mean_test = np.mean(testY)
    print('Accuracy: ', accuracy, '%')
    print('Coefficient of determination(r2):', r2, '%')
    print('Explained variance score:', EVS, '%')
    print('Mean absolute percentage error:', MAPE, '%')
    print('Mean absolute error:', MAE)
    print('Mean squared error:', MSE)
    print('Root mean squared error:', RMSE)
    print('Mean squared log error:', MSLE)
    print('Median absolute error:', MEDAE)
    print('Mean test:', Mean_test)
    print('Mean prediction:', Mean_pred)

    time_computation = (time() - start_time)
    #    print('Time:',round(time_computation), 'seconds')
    #    print(predY[:3])
    print('      Training done in ', time_computation, 'sec')
    print()
    results = {'r2[%]': r2, 'Accuracy[%]': accuracy, 'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE,
               'MSLE': MSLE, 'MEDAE': MEDAE, 'MAPE[%]': MAPE, 'EVS[%]': EVS,
               'RMSE_norm': RMSE_norm, 'MEDAE_norm': MEDAE_norm, 'MSLE_norm': MSLE_norm,
               'Time[sec]': time_computation, 'MeanPred': Mean_pred, }
    all_results.append(results)
    all_predictions.append(predY)
    return model, predY


# In[]: MLPRegressor
param_grid_MLPR = {
    "hidden_layer_sizes": [(10,), (20,), (50,), (100,), (200,)],
    "activation": ['relu', 'identity', 'logistic', 'tanh'],
    "solver": ['adam', 'lbfgs'],  # 'sgd'
    "learning_rate": ['constant', 'invscaling', 'adaptive'],
    "max_iter": [100]}
MLPR, predMLPR = optimize_model(MLPRegressor(), param_grid_MLPR, 'MLPRegressor')

# In[]: Support Vector Regressor
param_grid_SVR = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
    {'C': [1, 10, 100, 1000], 'degree': [2, 3], 'kernel': ['poly']}]
SVR, predSVR = optimize_model(SVR(), param_grid_SVR, 'SupportVectorRegressor')

# In[]: Kernel Ridge Regression
param_grid_KR = {"alpha": [0.01, 0.001, 0.0001, 0.00001],
                 'kernel': ['linear', 'rbf', 'poly'],
                 "gamma": [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
                 'degree': [2, 3]}
KR, predKR = optimize_model(KernelRidge(), param_grid_KR, 'KernelRidgeRegression')

# In[]: Gaussian Process Regressor
param_grid_GPR = {"kernel": [RBF(), RationalQuadratic(), ConstantKernel(), Matern()],
                  'alpha': [0.01, 0.001, 0.0001, 0.00001]}

GPR, predGPR = optimize_model(GaussianProcessRegressor(), param_grid_GPR, 'GaussianProcessRegressor')

# In[]: Gradient Boosting Regressor
param_grid_GBR = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                  'learning_rate': [10, 1, 0.1, 0.01],
                  'n_estimators': [80, 100, 120],
                  'max_depth': [3, 5, 7]}

GBR, predGBR = optimize_model(GradientBoostingRegressor(), param_grid_GBR, 'GradientBoostingRegressor')

# In[]: Decision Tree Regressor
param_grid_DTR = {'criterion': ['mse', 'mae'],
                  'max_depth': [3, 5, 7]}
DTR, predDTR = optimize_model(DecisionTreeRegressor(), param_grid_DTR, 'DecisionTreeRegressor')

# In[]: BayesianRidge
param_grid_BR = {'n_iter': [100, 200, 300, 400],
                 'tol': [1e-3, 1e-3, 1e-4],
                 'alpha_1': [1e-05, 1e-06, 1e-07],
                 'lambda_1': [1e-05, 1e-06, 1e-07],
                 'fit_intercept': [True, False],
                 'normalize': [True, False]}
BR, predBR = optimize_model(BayesianRidge(), param_grid_BR, 'BayesianRidge')

# In[]: KNeighbors Regressor

param_grid_KNR = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8],
                  'weights': ['uniform', 'distance']}

KNR, predKNR = optimize_model(KNeighborsRegressor(), param_grid_KNR, 'KNeighborsRegressor')

# In[]: AdaBoost Regressor

param_grid_ABR = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  'learning_rate': [10, 1, 0.1, 0.01],
                  'loss': ['linear', 'square', 'exponential']}

ABR, predABR = optimize_model(AdaBoostRegressor(), param_grid_ABR, 'AdaBoostRegressor')

