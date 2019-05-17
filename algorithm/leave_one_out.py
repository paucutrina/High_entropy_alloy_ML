from algorithms.Tools import *
from sklearn.model_selection import LeaveOneOut

subtract_ys_test = False
increased_ys = False
scale_data = False
estimated = None

if subtract_ys_test and increased_ys:
    raise Exception('If you want to subtract the YS, the increased_ys should be False')
elif subtract_ys_test and scale_data:
    raise Exception('If you want to subtract the YS, the scale_data should be False')

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)

if estimated:
    X, Y = data.get_estimated_data()
elif estimated is None:
    X, Y = data.get_pictures_data()
else:
    X, Y = data.get_no_estimated_data()

id_test = get_id_test(increased_ys, scale_data, estimated, comment='loo')

plots = Plots(id_test, increased_ys=increased_ys)

loo = LeaveOneOut()
loo.get_n_splits(X)


class TrainModels:
    def __init__(self):
        self.KNR = None
        self.BRR = None
        self.DTR = None
        self.GBR = GradientBoostingRegressor(learning_rate=0.1, loss='ls', max_depth=3, n_estimators=80)
        self.KRR = None
        self.GPR = None  # GaussianProcessRegressor(alpha=0.1,kernel=RationalQuadratic(alpha=1,length_scale=1))
        self.SVRe = None
        self.MLPR = None
        self.ABR = None


results_CV = []
for train_index, test_index in loo.split(X):
    trainX, testX = X[train_index], X[test_index]
    trainY, testY = Y[train_index], Y[test_index]
    insert_train_test = [trainX, testX, trainY, testY]
    models = Models(X, Y, TrainModels(), subtract_ys=subtract_ys_test, insert_train_test=insert_train_test)
    matrix_all_results, df_all_results, all_predictions = models.get_results()

    results_CV = np.append(results_CV, matrix_all_results[0, 7])

plots.mape_all_features(X, Y, results_CV)
