from algorithms.Tools import *
from algorithms.pictures import TrainModels

subtract_ys_test = False
increased_ys = False
scale_data = False
estimated = None
k_fold = False
input_pca = False

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)

if estimated:
    X, Y = data.get_estimated_data()
elif estimated is None:
    X, Y = data.get_pictures_data()
else:
    X, Y = data.get_no_estimated_data()

id_test = get_id_test(increased_ys, scale_data, estimated=estimated, comment='testsize')
plots = Plots(id_test,  increased_ys=increased_ys)

sizes_test = np.arange(0.05, 0.7, 0.05)
results_testSize = []
for test_size in sizes_test:
    models = Models(X, Y, TrainModels(), subtract_ys=subtract_ys_test, k_fold=k_fold, test_size=test_size)
    matrix_all_results, df_all_results, all_predictions = models.get_results()
    results_testSize = np.append(results_testSize, matrix_all_results)

results_testSize = np.reshape(results_testSize, (len(sizes_test), int(len(matrix_all_results[0]))))

sizes_train = np.arange(0.95 - max(sizes_test) + 0.05, 0.95 - min(sizes_test) + 0.05, 0.05)
sizes_train = sizes_train[::-1]
plots.evaltechniques_percentage(sizes_train, results_testSize, 'Train size')
points_plotted = np.column_stack((results_testSize[:, :1], results_testSize[:, 7:11]))

fig = plt.figure()
results_evaluation = []
for vertical_points in points_plotted:
    value_evaluation = (np.prod(vertical_points)) ** (1 / len(vertical_points))
    results_evaluation = np.append(results_evaluation, value_evaluation)
plt.scatter(sizes_train, results_evaluation)
plt.xlabel('Train size', fontsize=14)
plt.ylabel('Geometric mean [%]', fontsize=14)
plt.rcParams.update({'font.size': 14})
axes = plt.gca()
axes.set_ylim([0, 100])
plt.show()

