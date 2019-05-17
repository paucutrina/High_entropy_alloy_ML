from algorithms.Tools import *

increased_ys = False
scale_data = False
estimated = None

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)  # increased_ys=True

if estimated:
    X, Y = data.get_estimated_data()
elif estimated is None:
    X, Y = data.get_pictures_data()
else:
    X, Y = data.get_no_estimated_data()

# How to save plots
id_test = get_id_test(increased_ys, scale_data=scale_data, estimated=estimated, comment='LC')

plots = Plots(id_test, increased_ys=increased_ys)  # increased_ys=True

gbr = GradientBoostingRegressor(learning_rate=0.1, loss='ls', max_depth=3, n_estimators=80)
gpr = GaussianProcessRegressor(alpha=0.1, kernel=RationalQuadratic(alpha=1, length_scale=1))
if estimated is None:
    models = [gpr]
else:
    models = [gbr]
train_size = np.linspace(1.0, len(X)-len(X)*0.2, num=constant.num_samples_LC).astype(int)
for num_model, clf in enumerate(models):
    lc = LearningCurve(clf, X, Y, 'neg_mean_squared_error', train_size, id_test)
    gap, train_sizes = lc.plotLearningCurve()
    if gap[0] is not None:
        lc.plotGap(gap)
