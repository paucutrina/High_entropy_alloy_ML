from algorithms.Tools import *

subtract_ys_test = False
increased_ys = False
scale_data = False
estimated = False
k_fold = False
input_pca = False

if subtract_ys_test and increased_ys:
    raise Exception('If you want to subtract the YS, the increased_ys should be False')
elif scale_data:
    raise Exception('Can not be scaled the data')

data = Dataset(increased_ys=increased_ys, scale_data=scale_data)

if estimated:
    X, Y = data.get_estimated_data()
else:
    X, Y = data.get_no_estimated_data()

id_test = get_id_test(increased_ys, scale_data=scale_data, estimated=estimated, comment='extrapol_sfe')
plots = Plots(id_test, increased_ys=increased_ys)

energies_to_predict = ['127', '84', '72', '35']
results = []
for energy_to_predict in energies_to_predict:
    testX = []
    testY = []
    trainX = []
    trainY = []
    for num_sample, sample in enumerate(X):
        if energy_to_predict[0] == str(sample[0])[0] or \
                (str(sample[0])[0] == '6' and energy_to_predict[0] == '7'):
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

best_results = np.reshape(best_results, (int(len(energies_to_predict)), int(len(matrix_all_results[0]))))

plots.evaltechniques_percentage(energies_to_predict, best_results, 'Stacking fault energy [MPa]')