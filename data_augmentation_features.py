from algorithms.Tools import *

subtract_ys_test = False
increased_ys = False
estimated = True
scale_data = False
k_fold = True

if subtract_ys_test and increased_ys:
    raise Exception('If you want to subtract the YS, the increased_ys should be False')

data = Dataset(increased_ys=increased_ys)

if estimated:
    X, Y = data.get_estimated_data()
else:
    X, Y = data.get_no_estimated_data()
# How to save plots
id_test = get_id_test(increased_ys, scale_data=scale_data, estimated=estimated, comment='DA')

plots = Plots(id_test, increased_ys=increased_ys)  # increased_ys=True

x_added, y_added = plots.data_augmentation_ys_grainsize(X, Y, constant.amount_of_da, random_gs=False)

X = np.concatenate((X, x_added), axis=0)
Y = np.concatenate((Y, y_added), axis=0)

# Split energies
X, Y = split_energies(X, Y, gamma=constant.gamma_to_split)
X, Y = shuffle_list(X, Y)

models = Models(X, Y, StandardModels(), subtract_ys=subtract_ys_test, k_fold=k_fold)
matrix_all_results, df_all_results, all_predictions = models.get_results()

if subtract_ys_test:
    testY = convert_to_increased_YS(models.testX, models.testY)
else:
    testY = models.testY

# Yield Stress: Known vs Predicted
if k_fold is not True:
    plots.ys_test_pred(all_predictions, testY, 'GradientBoostingRegressor')

gbr = GradientBoostingRegressor(learning_rate=0.1, loss='ls', n_estimators=80, max_depth=2)
models = [gbr]
train_size = np.linspace(1.0, len(X)-len(X)*0.2, num=constant.num_samples_LC).astype(int)
for num_model, clf in enumerate(models):
    lc = LearningCurve(clf, X, Y,
                       num_model, 'neg_mean_squared_error', train_size, id_test)
    gap, train_sizes = lc.plotLearningCurve()
    if gap[0] is not None:
        lc.plotGap(gap)
