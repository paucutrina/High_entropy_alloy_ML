from algorithms.Tools import *
from algorithms.pictures import TrainModels

subtract_ys_test = False
increased_ys = False
scale_data = False
k_fold = False
test_type_pca = True  # False--> iteration number of components

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)
X, Y = data.get_pictures_data()

if test_type_pca:
    type_test = 'PCAcomp'
else:
    type_test = 'numfeat'

id_test = get_id_test(increased_ys, scale_data=scale_data, comment=type_test)
plots = Plots(id_test)  # increased_ys=True

if test_type_pca:
    array_numFeatures = np.arange(1, 84, 1)
else:
    number_nonRepeatedValues = []
    X_scale = preprocessing.scale(X)
    for row in X_scale:
        unique_value = len(np.unique(row))
        number_nonRepeatedValues = np.append(number_nonRepeatedValues, unique_value)
    max_useful_data = int(min(number_nonRepeatedValues))
    print('The minim number of non repeated values is:', max_useful_data)
    array_numFeatures = np.arange(5, max_useful_data, 100)

rmse = []
r2 = []
mape = []
mae = []
for num_features in array_numFeatures:
    if test_type_pca:
        X_new, eigenvalues, eigen_vectors = pca_input(num_features, copy.deepcopy(X))
    else:
        X_new = reduce_pixels_random(copy.deepcopy(X_scale), num_features)

    models = Models(X_new, Y, TrainModels(), subtract_ys=subtract_ys_test, k_fold=k_fold)
    matrix_all_results, df_all_results, all_predictions = models.get_results()

    rmse.append(matrix_all_results[:, 3])
    r2.append(matrix_all_results[:, 0])
    mape.append(matrix_all_results[:, 7])
    mae.append(matrix_all_results[:, 4])

rmse = np.reshape(rmse, (len(rmse), ))
r2 = np.reshape(r2, (len(r2),))
mape = np.reshape(mape, (len(mape), ))
mae = np.reshape(mae, (len(mae), ))

eval_techniques = [r2, rmse, mae, mape]
metrics = ['R2[%]', 'RMSE', 'MAE', 'MAPE[%]']
for num, eval_tech in enumerate(eval_techniques):
    plt.figure()
    plt.title('PCA(n)')
    plt.xlabel('number of components')
    plt.ylabel(metrics[num])
    plt.scatter(array_numFeatures, eval_tech, s=5)
    z = np.polyfit(array_numFeatures, eval_tech, 10)
    p = np.poly1d(z)
    plt.plot(array_numFeatures, p(array_numFeatures), ':', linewidth=0.5)
    plt.show()

