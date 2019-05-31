from new_ann import run_ann
from process_file import process_new_file
from process_data import pickle_data
import os


trial_num = 6
# folders = ['anns', 'confusion', 'loss', 'output']
# print('creating folders\n')
# for folder in folders:
#     dirName = 'trials' + str(trial_num) + '/' + folder
#     # Create target directory & all intermediate directories if don't exists
#     os.makedirs(dirName)

# if you are testing the ann, run this
num_units = 6
num_hidden_layers = 1
num_epochs = 20
batch_size = 200
test_train_split = .4
while num_epochs <= 100:
    while batch_size <= 400:
        run_ann('pickle','mean_sd', num_units, num_hidden_layers, num_epochs, batch_size, test_train_split, trial_num)
        batch_size += 50
    num_epochs += 20
    batch_size = 200

# if you are using an ANN to process a new file, use this
# file_name = 'track2'
# path = 'C:\\Users\\rjoyh\\Desktop\\machine learning\\trials' + str(trial_num) + '\\anns\\'
# sub_path = 'trials' + str(trial_num) + '\\anns\\'
# files = []
# # r=root, d=directories, f = files
# for r, d, f in os.walk(path):
#     for file in f:
#         if '.h5' in file:
#             files.append(file)
#
# for f in files:
#     process_new_file(file_name, sub_path + f, trial_num)

# if you have new data, run this
# pickle_data('mean_sd',6)
