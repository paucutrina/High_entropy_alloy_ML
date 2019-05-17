from algorithms.Tools import *

one_sfe = False
subtract_ys_test = False
increased_ys = False
scale_data = False
estimated = False
k_fold = True
input_pca = False

if subtract_ys_test and increased_ys:
    raise Exception('If you want to subtract the YS, the increased_ys should be False')
elif subtract_ys_test and scale_data:
    raise Exception('If you want to subtract the YS, the scale_data should be False')

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)

if estimated:
    X, Y = data.get_estimated_data()
else:
    X, Y = data.get_no_estimated_data()

# Split energies
if one_sfe:
    X, Y = split_energies(X, Y, gamma=constant.gamma_to_split)
    X, Y = shuffle_list(X, Y)

if input_pca:
    X, eigenvalues, eigen_vectors = pca_input(3, X)  # PCA as input

# How to save plots
id_test = get_id_test(increased_ys, scale_data, estimated=estimated, input_pca=input_pca, comment='features')

plots = Plots(id_test, increased_ys=increased_ys)

# YS vs Grain size for each gamma
# if input_pca is False and scale_data is False:
#     plots.ys_grainsize(X, Y)

# Plot features without PCA
# plots.data_features(X, Y)

# Plot features with PCA(3 components)
# plots.pca_three_components(X, Y)

# Plot features with PCA(n components)
# plots.pca_ncomponents(X, Y, 2)

# Other techniques
# plots.unsupervised_models(X, Y)

models = Models(X, Y, StandardModels(), subtract_ys=subtract_ys_test, k_fold=k_fold)
matrix_all_results, df_all_results, all_predictions = models.get_results()
print(matrix_all_results)
if subtract_ys_test:
    testY = convert_to_increased_YS(models.testX, models.testY)
else:
    testY = models.testY

if k_fold is not True:
    # Yield Stress: Known vs Predicted
    plots.ys_test_pred(all_predictions, testY, 'GradientBoostingRegressor')

print('R2=', matrix_all_results[:, 0])
print('MAE=', matrix_all_results[:, 4])
print('RMSE=', matrix_all_results[:, 3])
print('MAPE=', matrix_all_results[:, 7])
