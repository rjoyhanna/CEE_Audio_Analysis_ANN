from new_ann_obj import ANN
from file_helper import create_trial_folder


# DATA = ['data001.pickle', 'data002.pickle', 'data003.pickle']
# UNITS = [3, 6]
# LAYERS = [1, 3]
# EPOCHS = [20]
# BATCH_SIZE = [50, 300]
# SPLIT = [.4]

DATA = ['data002.pickle', 'data003.pickle', 'data006.pickle', 'data007.pickle']
UNITS = [6]
LAYERS = [1, 2]
EPOCHS = [30]
BATCH_SIZE = [300]
SPLIT = [.4]


class Trial:
    def __init__(self, trial_num, save_graphs=True, save_anns=False):
        best_acc = 0
        best_ann = None
        self.trial_num = trial_num
        self.total_anns = len(DATA) * len(UNITS) * len(LAYERS) * len(EPOCHS) * len(BATCH_SIZE) * len(SPLIT)
        i = 0

        create_trial_folder(trial_num)

        for data in DATA:
            for num_units in UNITS:
                for num_hidden_layers in LAYERS:
                    for num_epochs in EPOCHS:
                        for batch_size in BATCH_SIZE:
                            for test_train_split in SPLIT:
                                i += 1
                                print('\n\nbuilding ANN {} of {}:\nData: {}\tUnits: {}\tLayers: {}\t'
                                      'Epochs: {}\tBatch: {}\tSplit: {}'.format(i, self.total_anns, data, num_units,
                                                                                num_hidden_layers, num_epochs,
                                                                                batch_size, test_train_split))
                                curr_ann = ANN(data, num_units, num_hidden_layers, num_epochs, batch_size,
                                               test_train_split, self.trial_num)
                                curr_acc = (curr_ann.hist.history['acc'][len(curr_ann.hist.history['acc']) - 1]) * 100
                                if save_anns:
                                    curr_ann.save_obj()
                                if save_graphs:
                                    num_labelled_0, num_labelled_1, num_labelled_2, \
                                    num_labelled_3 = curr_ann.create_graphs()
                                    if curr_acc > best_acc:
                                        if num_labelled_0 != 0 and num_labelled_1 != 0 and num_labelled_2 != 0 or num_labelled_3 != 0:
                                            best_acc = curr_acc
                                            best_ann = curr_ann
                                else:
                                    best_acc = curr_acc
                                    best_ann = curr_ann

        self.best_ann = best_ann
        self.best_acc = best_acc


trial11 = Trial(11)
print(trial11.best_acc)
