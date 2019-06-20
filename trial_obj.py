from new_ann_obj import ANN
from file_helper import get_valid_data_files


DATA = ['data001.pickle', 'data002.pickle']
UNITS = [3, 6, 9, 20]
LAYERS = [1, 2, 5]
EPOCHS = [20, 100]
BATCH_SIZE = [50, 100, 200, 500]
SPLIT = [.3, .4]


class Trial:
    def __init__(self, trial_num, save_graphs=True, save_anns=False):
        best_acc = 0
        best_ann = None
        self.trial_num = trial_num
        self.total_anns = len(DATA) * len(UNITS) * len(LAYERS) * len(EPOCHS) * len(BATCH_SIZE) * len(SPLIT)
        i = 0

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
                                if curr_acc > best_acc:
                                    best_acc = curr_acc
                                    best_ann = curr_ann
                                if save_anns:
                                    curr_ann.save_obj()
                                if save_graphs:
                                    curr_ann.create_graphs()

        self.best_ann = best_ann
        self.best_acc = best_acc
