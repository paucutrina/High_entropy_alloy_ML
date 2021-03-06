from algorithms.Tools import *

one_sfe = False
subtract_ys_test = False
increased_ys = False
scale_data = False
estimated = False
k_fold = False
input_pca = False

if estimated:
    raise Exception('Can not be estimated')
elif scale_data:
    raise Exception('Can not be scaled the data')

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)

X, Y = data.get_no_estimated_data()

id_test = get_id_test(increased_ys, scale_data=scale_data, estimated=estimated, comment='extrapol_grain')
plots = Plots(id_test, increased_ys=increased_ys)

grains_to_predict = [0.252, 0.5, 1., 2., 3.19, 5.82, 11.27, 8., 4.5]
results = []
for num_grain, grain_size in enumerate(grains_to_predict):
    testX = []
    testY = []
    trainX = []
    trainY = []
    for num_sample, sample in enumerate(X):
        if grain_size == sample[2]:
            testX = np.append(testX, sample)
            testY = np.append(testY, Y[num_sample])
        else:
            trainX = np.append(trainX, sample)
            trainY = np.append(trainY, Y[num_sample])

    trainX = np.reshape(trainX, (int(len(trainX) / int(len(X[0]))), int(len(X[0]))))
    testX = np.reshape(testX, (int(len(testX) / int(len(X[0]))), int(len(X[0]))))
    np.random.seed(0)

    trainX, trainY = shuffle_list(trainX, trainY)
    testX, testY = shuffle_list(testX, testY)
    insert_train_test = [trainX, testX, trainY, testY]
    models = Models(X, Y, StandardModels(), subtract_ys=subtract_ys_test, k_fold=k_fold,
                    insert_train_test=insert_train_test)
    matrix_all_results, df_all_results, all_predictions = models.get_results()
    results.append(df_all_results)

best_results = []
for result in results:
    order_result = result.sort_values(by=['r2[%]'])
    best_results = np.append(best_results, order_result.values[-1].tolist())

best_results = np.reshape(best_results, (int(len(matrix_all_results)), int(len(matrix_all_results[0]))))

plots.evaltechniques_percentage(grains_to_predict[:-2], best_results[:-2], 'Region Size [nm]')