import csv
import numpy as np
import os
import re
import pickle
from time import time


def save_variable(variable, name):
    f = open(name + '.pckl', 'wb')
    pickle.dump(variable, f)
    f.close()

path = "data/dat_files/"
folders = os.listdir(path)
insert_diameter = True
insert_simulation = True
gamma_list = []
all_pictures = []
all_pictures_id = []
for folder in folders:
    print(folder)
    gamma = int(re.search(r'\d+', folder).group())
    gamma_list.append(gamma)
    path_folder = path + folder + '/'
    files = os.listdir(path_folder)
    diameter_list = []
    pictures = []
    pictures_id = []
    for file in files:
        print(file)
        diameter = int(re.search(r'\d+', file).group())
        diameter_list.append(diameter)
        path_files = path_folder + file + '/'
        samples = os.listdir(path_files)
        simulation_list = []
        for sample in samples:
            print(sample)
            simulation_list.append(sample)
            obtain_id = sample.split('_')
            gamma_value = int(re.search(r'\d+', obtain_id[0]).group())
            diameter_value = int(re.search(r'\d+', obtain_id[1]).group())
            simulation_value = int(re.search(r'\d+', obtain_id[2]).group())
            path_samples = path_files + sample
            samples_list = []
            vector_position_intensity = []
            with open(path_samples) as f:
                reader = csv.reader(f, delimiter="\t")
                num_row = 0
                start_time = time()
                for num_row, row in enumerate(reader):
                    data_row = row[0].split(' ')
                    data_row_clean = ' '.join(data_row).split()
                    samples_list.append(data_row_clean)
                    if num_row == 0:
                        yield_stress = samples_list[0][2]
                    elif num_row == 1:
                        pass
                    else:
                        values_picture = row[0].split(' ')
                        x_position = values_picture[0]
                        y_position = values_picture[3]
                        intensity = values_picture[16]
                        vector_position_intensity = np.append(vector_position_intensity, x_position)
                        vector_position_intensity = np.append(vector_position_intensity, y_position)
                        vector_position_intensity = np.append(vector_position_intensity, intensity)
                matrix_position_intensity = np.reshape(vector_position_intensity,
                                                       (int(len(vector_position_intensity) / 3), 3))
                matrix_picture = np.reshape(matrix_position_intensity[:, 2], (256, 256))
                vector_picture = matrix_position_intensity[:, 2]
                matrix_picture = matrix_picture.astype(np.float)
                info_pictures = np.array([sample, yield_stress])
                pictures_id.append(info_pictures)
                all_pictures_id.append(info_pictures)
                pictures.append(vector_picture)
                all_pictures.append(vector_picture)
                # save_variable(matrix_picture,
                #              'variables/Matrix_' + 'isf' + str(gamma) + '_d' + str(diameter_value) + 'nm_' + str(
                #                  simulation_value))
                # save_variable(vector_picture,
                #              'variables/Vector_' + 'isf' + str(gamma) + '_d' + str(diameter_value) + 'nm_' + str(
                #                  simulation_value))
                print('Time:', (time() - start_time) / 60)
                print('######################')

    # save_variable(pictures, 'variables/allGamma_' + 'isf' + str(gamma))
    # save_variable(pictures_id, 'variables/all_idGamma_' + 'isf' + str(gamma))

save_variable(all_pictures, 'variables/allPictures')
save_variable(all_pictures_id, 'variables/all_idPictures')
print('Total Time:', (time() - start_time) / 60)
