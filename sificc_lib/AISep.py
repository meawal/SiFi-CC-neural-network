import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from sificc_lib import utils
import pickle as pkl
import datetime as dt
import uproot
from sificc_lib import AI, MyCallback
        
class AISep(AI):
    def __init__(self, data_model, model_name=None):
        '''Initializing an instance of SiFi-CC Neural Network
        '''
        super().__init__(data_model, model_name)
        self._weight_e_tcluster_fltr = 1
        self._weight_p_tcluster_fltr = 1
        self._weight_e_pcluster_fltr = 1
        self._weight_p_pcluster_fltr = 1
        self._weight_e_ecluster_fltr = 1
        self._weight_p_ecluster_fltr = 1
        self._weight_type_fltr = 1
        self._weight_pos_x_fltr = 1
        self._weight_pos_y_fltr = 1
        self._weight_pos_z_fltr = 1
        self._weight_energy_fltr = 1
        
        
    def create_model(self, conv_layers=[], classifier_layers=[], dense_layers=[],
                     type_layers=[], pos_layers=[], energy_layers=[], 
                     base_l2=0, limbs_l2=0, 
                     conv_dropouts=[], activation='relu', 
                     pos_loss=None, energy_loss=None):
        if len(conv_dropouts) == 0:
            conv_dropouts = [0] * len(conv_layers)
        assert len(conv_dropouts) == len(conv_layers)
        
        ###### input layer ######
        feed_in = keras.Input(shape=self.data.get_features(0,1).shape[1:], name='inputs')
        
        ###### type nn layers ######
        # convolution layers
        cnv = feed_in
        for i in range(len(conv_layers)):
            cnv = keras.layers.Conv1D(conv_layers[i], 
                                    kernel_size = self.data.cluster_size if i == 0 else 1, 
                                    strides = self.data.cluster_size if i == 0 else 1, 
                                    activation = activation, 
                                    kernel_regularizer = keras.regularizers.l2(base_l2), 
                                    padding = 'valid', name='typ_conv_{}'.format(i+1))(cnv)
            if conv_dropouts[i] != 0:
                cnv = keras.layers.Dropout(dropouts[i])(cnv)
                
        if len(conv_layers) >= 1:
            cnv = keras.layers.Flatten(name='typ_flatting')(cnv)
            
        # clusters classifier layers
        cls = cnv
        for i in range(len(classifier_layers)):
            cls = keras.layers.Dense(classifier_layers[i], activation=activation, 
                                            kernel_regularizer=keras.regularizers.l2(base_l2), 
                                            name='typ_dense_cluster_{}'.format(i+1))(cls)
            
        # e/p clusters classifiers
        typ_e_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='typ_e_cluster')(cls)
        typ_p_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='typ_p_cluster')(cls)
        
        # get the hardmax of clusters classifier
        e_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='typ_e_hardmax')(typ_e_cluster)
        p_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='typ_p_hardmax')(typ_p_cluster)
        
        # joining outputs
        base_layer = keras.layers.Concatenate(axis=-1, name='typ_join_layer')(
                                            [cnv, e_cluster_1_hot, p_cluster_1_hot])
        
        # event type layers
        typ = base_layer
        for i in range(len(type_layers)):
            typ = keras.layers.Dense(type_layers[i], activation=activation, 
                                   kernel_regularizer = keras.regularizers.l2(limbs_l2), 
                                   name='typ_dense_type_{}'.format(i+1))(typ)
            
        event_type = keras.layers.Dense(1, activation='sigmoid', name='typ_type')(typ)
        
        ###### position nn layers ######
        # convolution layers
        cnv = feed_in
        for i in range(len(conv_layers)):
            cnv = keras.layers.Conv1D(conv_layers[i], 
                                    kernel_size = self.data.cluster_size if i == 0 else 1, 
                                    strides = self.data.cluster_size if i == 0 else 1, 
                                    activation = activation, 
                                    kernel_regularizer = keras.regularizers.l2(base_l2), 
                                    padding = 'valid', name='pos_conv_{}'.format(i+1))(cnv)
            if conv_dropouts[i] != 0:
                cnv = keras.layers.Dropout(dropouts[i])(cnv)
                
        if len(conv_layers) >= 1:
            cnv = keras.layers.Flatten(name='pos_flatting')(cnv)
            
        # clusters classifier layers
        cls = cnv
        for i in range(len(classifier_layers)):
            cls = keras.layers.Dense(classifier_layers[i], activation=activation, 
                                            kernel_regularizer=keras.regularizers.l2(base_l2), 
                                            name='pos_dense_cluster_{}'.format(i+1))(cls)
            
        # e/p clusters classifiers
        pos_e_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='pos_e_cluster')(cls)
        pos_p_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='pos_p_cluster')(cls)
        
        # get the hardmax of clusters classifier
        e_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='pos_e_hardmax')(pos_e_cluster)
        p_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='pos_p_hardmax')(pos_p_cluster)
        
        # joining outputs
        base_layer = keras.layers.Concatenate(axis=-1, name='pos_join_layer')(
                                            [cnv, e_cluster_1_hot, p_cluster_1_hot])
        
        # event position
        pos = base_layer
        for i in range(len(pos_layers)):
            pos = keras.layers.Dense(pos_layers[i], activation=activation, 
                                     kernel_regularizer= keras.regularizers.l2(limbs_l2), 
                                     name='pos_dense_{}'.format(i+1))(pos)
            
        pos_x = keras.layers.Dense(2, activation=None, name='pos_x')(pos)
        pos_y = keras.layers.Dense(2, activation=None, name='pos_y')(pos)
        pos_z = keras.layers.Dense(2, activation=None, name='pos_z')(pos)
        
        ###### energy nn layers ######
        # convolution layers
        cnv = feed_in
        for i in range(len(conv_layers)):
            cnv = keras.layers.Conv1D(conv_layers[i], 
                                    kernel_size = self.data.cluster_size if i == 0 else 1, 
                                    strides = self.data.cluster_size if i == 0 else 1, 
                                    activation = activation, 
                                    kernel_regularizer = keras.regularizers.l2(base_l2), 
                                    padding = 'valid', name='enrg_conv_{}'.format(i+1))(cnv)
            if conv_dropouts[i] != 0:
                cnv = keras.layers.Dropout(dropouts[i])(cnv)
                
        if len(conv_layers) >= 1:
            cnv = keras.layers.Flatten(name='enrg_flatting')(cnv)
            
        # clusters classifier layers
        cls = cnv
        for i in range(len(classifier_layers)):
            cls = keras.layers.Dense(classifier_layers[i], activation=activation, 
                                            kernel_regularizer=keras.regularizers.l2(base_l2), 
                                            name='enrg_dense_cluster_{}'.format(i+1))(cls)
            
        # e/p clusters classifiers
        enrg_e_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='enrg_e_cluster')(cls)
        enrg_p_cluster = keras.layers.Dense(self.data.clusters_limit, activation='softmax', 
                                       name='enrg_p_cluster')(cls)
        
        # get the hardmax of clusters classifier
        e_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='enrg_e_hardmax')(enrg_e_cluster)
        p_cluster_1_hot = keras.layers.Lambda(
            lambda x: K.one_hot(K.argmax(x), self.data.clusters_limit), 
            name='enrg_p_hardmax')(enrg_p_cluster)
        
        # joining outputs
        base_layer = keras.layers.Concatenate(axis=-1, name='enrg_join_layer')(
                                            [cnv, e_cluster_1_hot, p_cluster_1_hot])
        
        # event energy 
        enrg = base_layer
        for i in range(len(energy_layers)):
            enrg = keras.layers.Dense(energy_layers[i], activation=activation, 
                                      kernel_regularizer= keras.regularizers.l2(limbs_l2), 
                                      name='enrg_dense_{}'.format(i+1))(enrg)
            
        energy = keras.layers.Dense(2, activation=None, name='enrg_energy')(enrg)
        
        ###### building the model ######
        self.model = keras.models.Model(feed_in, [typ_e_cluster, typ_p_cluster, 
                                                  pos_e_cluster, pos_p_cluster, 
                                                  enrg_e_cluster, enrg_p_cluster, 
                                                  event_type, 
                                                  pos_x, pos_y, pos_z, energy])
        self.history = None
        self.model.summary()
        
    def compile_model(self, learning_rate=0.0003):
        self.model.compile(optimizer= keras.optimizers.Nadam(learning_rate), 
                           loss = {
                               'typ_e_cluster': self._e_cluster_loss,
                               'typ_p_cluster': self._p_cluster_loss,
                               'pos_e_cluster': self._e_cluster_loss,
                               'pos_p_cluster': self._p_cluster_loss,
                               'enrg_e_cluster': self._e_cluster_loss,
                               'enrg_p_cluster': self._p_cluster_loss,
                               'typ_type' : self._type_loss,
                               'pos_x': self._pos_loss,
                               'pos_y': self._pos_loss,
                               'pos_z': self._pos_loss,
                               'enrg_energy': self._energy_loss,
                           }, 
                           metrics = {
                               'typ_e_cluster': [self._cluster_accuracy],
                               'typ_p_cluster': [self._cluster_accuracy],
                               'pos_e_cluster': [self._cluster_accuracy],
                               'pos_p_cluster': [self._cluster_accuracy],
                               'enrg_e_cluster': [self._cluster_accuracy],
                               'enrg_p_cluster': [self._cluster_accuracy],
                               'typ_type' : [self._type_accuracy, self._type_tp_rate],
                               'pos_x': [],
                               'pos_y': [],
                               'pos_z': [],
                               'enrg_energy': [],
                           }, 
                           loss_weights = {
                               'typ_e_cluster': self.weight_e_cluster * self._weight_e_tcluster_fltr,
                               'typ_p_cluster': self.weight_p_cluster * self._weight_p_tcluster_fltr,
                               'pos_e_cluster': self.weight_e_cluster * self._weight_e_pcluster_fltr,
                               'pos_p_cluster': self.weight_p_cluster * self._weight_p_pcluster_fltr,
                               'enrg_e_cluster': self.weight_e_cluster * self._weight_e_ecluster_fltr,
                               'enrg_p_cluster': self.weight_p_cluster * self._weight_p_ecluster_fltr,
                               'typ_type' : self.weight_type * self._weight_type_fltr,
                               'pos_x': self.weight_pos_x * self._weight_pos_x_fltr,
                               'pos_y': self.weight_pos_y * self._weight_pos_y_fltr,
                               'pos_z': self.weight_pos_z * self._weight_pos_z_fltr,
                               'enrg_energy': self.weight_energy * self._weight_energy_fltr,
                           })
    
    def activate_all(self):
        for layer in self.model.layers:
            layer.trainable = True
        
        self._weight_e_tcluster_fltr = 1
        self._weight_p_tcluster_fltr = 1
        self._weight_e_pcluster_fltr = 1
        self._weight_p_pcluster_fltr = 1
        self._weight_e_ecluster_fltr = 1
        self._weight_p_ecluster_fltr = 1
        self._weight_type_fltr = 1
        self._weight_pos_x_fltr = 1
        self._weight_pos_y_fltr = 1
        self._weight_pos_z_fltr = 1
        self._weight_energy_fltr = 1
        
        self.compile_model()
        
    def activate_type_training(self):
        for layer in self.model.layers:
            if layer.name.startswith('typ_'):
                layer.trainable = True
                print(f'activating {layer.name}')
            else:
                layer.trainable = False
        
        self._weight_e_tcluster_fltr = 1
        self._weight_p_tcluster_fltr = 1
        self._weight_e_pcluster_fltr = 0
        self._weight_p_pcluster_fltr = 0
        self._weight_e_ecluster_fltr = 0
        self._weight_p_ecluster_fltr = 0
        self._weight_type_fltr = 1
        self._weight_pos_x_fltr = 0
        self._weight_pos_y_fltr = 0
        self._weight_pos_z_fltr = 0
        self._weight_energy_fltr = 0
        
        self.compile_model()
        
    def activate_position_training(self):
        for layer in self.model.layers:
            if layer.name.startswith('pos_'):
                layer.trainable = True
                print(f'activating {layer.name}')
            else:
                layer.trainable = False
                
        self._weight_e_tcluster_fltr = 0
        self._weight_p_tcluster_fltr = 0
        self._weight_e_pcluster_fltr = 1
        self._weight_p_pcluster_fltr = 1
        self._weight_e_ecluster_fltr = 0
        self._weight_p_ecluster_fltr = 0
        self._weight_type_fltr = 0
        self._weight_pos_x_fltr = 1
        self._weight_pos_y_fltr = 1
        self._weight_pos_z_fltr = 1
        self._weight_energy_fltr = 0
        
        self.compile_model()
        
    def activate_energy_training(self):
        for layer in self.model.layers:
            if layer.name.startswith('enrg_'):
                layer.trainable = True
                print(f'activating {layer.name}')
            else:
                layer.trainable = False
                
        self._weight_e_tcluster_fltr = 0
        self._weight_p_tcluster_fltr = 0
        self._weight_e_pcluster_fltr = 0
        self._weight_p_pcluster_fltr = 0
        self._weight_e_ecluster_fltr = 1
        self._weight_p_ecluster_fltr = 1
        self._weight_type_fltr = 0
        self._weight_pos_x_fltr = 0
        self._weight_pos_y_fltr = 0
        self._weight_pos_z_fltr = 0
        self._weight_energy_fltr = 1
        
        self.compile_model()
        
    
    def predict(self, data, denormalize=False, verbose=0):
        pred = self.model.predict(data)
        pred = np.concatenate([np.round(pred[6]), 
                pred[10], 
                pred[7][:,[0]], pred[8][:,[0]], pred[9][:,[0]], 
                pred[7][:,[1]], pred[8][:,[1]], pred[9][:,[1]], 
               ], axis=1)
        if denormalize:
            pred = self.data._denormalize_targets(pred)
            
        return pred
    
    def plot_training_loss(self, mode='eff', skip=0, smooth=True, summed_loss=True):
        def plot_line(ax, key, label, style, color):
            metric = self.history[key][skip:]
            metric = utils.exp_ma(metric, factor=.8) if smooth else metric
            ax.plot(np.arange(1,len(metric)+1)+skip, metric, style, label=label, color=color)
            
        def plot_metric(ax, key, label, color):
            if key in self.history:
                plot_line(ax, key, label, '-', color)
                if 'val_' + key in self.history:
                    plot_line(ax, 'val_' + key, None, '--', color)

        fig, ax1 = plt.subplots(figsize=(12,4))
        ax2 = ax1.twinx()  

        color = 'tab:blue'
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid()
        
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Precision', color=color)  
        ax1.tick_params(axis='y', labelcolor=color)
        
        if summed_loss:
            plot_metric(ax2, 'loss', 'Loss', 'tab:blue')
        
        if mode == 'acc':
            plot_metric(ax1, 'typ_type__type_accuracy', 'Type accuracy', 'tab:pink')
            plot_metric(ax1, 'typ_type__type_tp_rate', 'TP rate', 'tab:purple')
            plot_metric(ax1, 'typ_e_cluster__cluster_accuracy', 'e cluster acc', 'tab:red')
            plot_metric(ax1, 'typ_p_cluster__cluster_accuracy', 'p cluster acc', 'tab:brown')
        elif mode == 'eff':
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
            plot_metric(ax1, 'pur', 'Purity', 'tab:orange')
        elif mode == 'loss':
            plot_metric(ax1, 'typ_e_cluster_loss', 'Cluster e', 'tab:pink')
            plot_metric(ax1, 'typ_p_cluster_loss', 'Cluster p', 'tab:orange')
            plot_metric(ax1, 'pos_x_loss', 'Pos x', 'tab:brown')
            plot_metric(ax1, 'pos_y_loss', 'Pos y', 'tab:red')
            plot_metric(ax1, 'pos_z_loss', 'Pos z', 'tab:purple')
            plot_metric(ax1, 'typ_type_loss', 'Type', 'tab:orange')
            plot_metric(ax1, 'enrg_energy_loss', 'Energy', 'tab:cyan')
        elif mode == 'loss-cluster':
            plot_metric(ax2, 'typ_e_cluster_loss', 'Cluster e', 'tab:pink')
            plot_metric(ax2, 'typ_p_cluster_loss', 'Cluster p', 'tab:orange')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        elif mode == 'loss-pos':
            plot_metric(ax2, 'pos_x_loss', 'Pos x', 'tab:brown')
            plot_metric(ax2, 'pos_y_loss', 'Pos y', 'tab:red')
            plot_metric(ax2, 'pos_z_loss', 'Pos z', 'tab:purple')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        elif mode == 'loss-type':
            plot_metric(ax2, 'typ_type_loss', 'Type', 'tab:orange')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        elif mode == 'loss-energy':
            plot_metric(ax2, 'enrg_energy_loss', 'Energy', 'tab:cyan')
            plot_metric(ax1, 'eff', 'Effeciency', 'tab:pink')
        else:
            raise Exception('Invalid mode')
            
        ax1.plot([], '--', color='tab:gray', label='Validation')
                
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()

        ax2.yaxis.set_label_position('left')
        ax2.yaxis.tick_left()

        fig.legend(loc='upper left')
        fig.tight_layout()
        plt.show()
        
    def evaluate(self):
        [loss, e_cluster_loss, p_cluster_loss, _,_,_,_, type_loss, 
         pos_x_loss, pos_y_loss, pos_z_loss, energy_loss, 
         e_cluster__cluster_accuracy, p_cluster__cluster_accuracy, _,_,_,_,
         type__type_accuracy, type__type_tp_rate] = self.model.evaluate(
            self.data.test_x, self.data.test_y, verbose=0)
        
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        l_matches = self._find_matches(y_true, y_pred, keep_length=False)
        effeciency = np.mean(l_matches)
        purity = np.sum(l_matches) / np.sum(y_pred[:,0])
        precision = np.sum(y_pred[:,0] * y_true[:,0]) / np.sum(y_pred[:,0])
        recall = np.sum(y_pred[:,0] * y_true[:,0]) / np.sum(y_true[:,0])
        
        identified_events = np.array(self._find_matches(y_true, y_pred, keep_length=True, mask=[1]+([0]*8))).astype(bool)
        y_pred = self.data._denormalize_targets(y_pred[identified_events])
        y_true = self.data._denormalize_targets(y_true[identified_events])
        enrg = np.abs(y_true[:,1:3] - y_pred[:,1:3])
        enrg = enrg.ravel()
        mean_enrg = np.mean(enrg)
        std_enrg = np.std(enrg)
        euc = y_true[:,3:9] - y_pred[:,3:9]
        euc = euc.reshape((-1,3))
        euc = np.power(euc, 2)
        euc = np.sqrt(np.sum(euc, axis=1))
        mean_euc = np.mean(euc)
        std_euc = np.std(euc)
                
        print('AI model')
        print('  Loss:       {:8.5f}'.format(loss))
        print('    -Type:        {:8.5f} * {:5.2f} = {:7.5f}'.format(type_loss, self.weight_type, 
                                                                 type_loss * self.weight_type))
        print('    -Pos X:       {:8.5f} * {:5.2f} = {:7.5f}'.format(pos_x_loss, self.weight_pos_x, 
                                                                 pos_x_loss * self.weight_pos_x))
        print('    -Pos Y:       {:8.5f} * {:5.2f} = {:7.5f}'.format(pos_y_loss, self.weight_pos_y, 
                                                                 pos_y_loss * self.weight_pos_y))
        print('    -Pos Z:       {:8.5f} * {:5.2f} = {:7.5f}'.format(pos_z_loss, self.weight_pos_z, 
                                                                 pos_z_loss * self.weight_pos_z))
        print('    -Energy:      {:8.5f} * {:5.2f} = {:7.5f}'.format(energy_loss, self.weight_energy, 
                                                                 energy_loss * self.weight_energy))
        print('    -Cls e:       {:8.5f} * {:5.2f} = {:7.5f}'.format(e_cluster_loss, self.weight_e_cluster, 
                                                                 e_cluster_loss * self.weight_e_cluster))
        print('    -Cls p:       {:8.5f} * {:5.2f} = {:7.5f}'.format(p_cluster_loss, self.weight_p_cluster, 
                                                                 p_cluster_loss * self.weight_p_cluster))
        print('  Accuracy:    {:8.5f}'.format(type__type_accuracy))
        print('    -Precision:   {:8.5f}'.format(precision))
        print('    -Recall:      {:8.5f}'.format(recall))
        print('    -Cls e rate:  {:8.5f}'.format(e_cluster__cluster_accuracy))
        print('    -Cls p rate:  {:8.5f}'.format(p_cluster__cluster_accuracy))
        print('  Efficiency:  {:8.5f}'.format(effeciency))
        print('  Purity:      {:8.5f}'.format(purity))
        print('  Euc mean:    {:8.5f}'.format(mean_euc))
        print('  Euc std:     {:8.5f}'.format(std_euc))
        print('  Energy mean: {:8.5f}'.format(mean_enrg))
        print('  Energy std:  {:8.5f}'.format(std_enrg))
        
        
        y_pred = self.data.reco_test
        y_true = self.data.test_row_y
        l_matches = self._find_matches(y_true, y_pred, keep_length=False)
        effeciency = np.mean(l_matches)
        purity = np.sum(l_matches) / np.sum(y_pred[:,0])
        accuracy = self._type_accuracy(y_true[:,0], y_pred[:,0]).numpy()
        tp_rate = self._type_tp_rate2(y_true[:,0], y_pred[:,0]).numpy()
        
        identified_events = np.array(self._find_matches(y_true, y_pred, keep_length=True, mask=[1]+([0]*8))).astype(bool)
        y_pred = self.data._denormalize_targets(y_pred[identified_events])
        y_true = self.data._denormalize_targets(y_true[identified_events])
        enrg = np.abs(y_true[:,1:3] - y_pred[:,1:3])
        enrg = enrg.ravel()
        mean_enrg = np.mean(enrg)
        std_enrg = np.std(enrg)
        euc = y_true[:,3:9] - y_pred[:,3:9]
        euc = euc.reshape((-1,3))
        euc = np.power(euc, 2)
        euc = np.sqrt(np.sum(euc, axis=1))
        mean_euc = np.mean(euc)
        std_euc = np.std(euc)
        
        print('\nReco')
        print('  Accuracy:    {:8.5f}'.format(accuracy))
        print('    -TP rate:     {:8.5f}'.format(tp_rate))
        print('  Efficiency:  {:8.5f}'.format(effeciency))
        print('  Purity:      {:8.5f}'.format(purity))
        print('  Euc mean:    {:8.5f}'.format(mean_euc))
        print('  Euc std:     {:8.5f}'.format(std_euc))
        print('  Energy mean: {:8.5f}'.format(mean_enrg))
        print('  Energy std:  {:8.5f}'.format(std_enrg))

    def plot_diff(self, mode='type-match', add_reco=True, focus=False):
        y_pred = self.predict(self.data.test_x)
        y_true = self.data.test_row_y
        y_reco = self.data.reco_test

        if mode == 'all-match':
            mask = None
        elif mode == 'pos-match':
            mask = [1,0,0,1,1,1,1,1,1]
        elif mode == 'type-match':
            mask = [1] + ([0] * 8)
        elif mode == 'miss':
            mask = None
        else:
            raise Exception('mode {} not recognized'.format(mode))

        l_matches = np.array(self._find_matches(y_true, y_pred, mask, keep_length = True)).astype(bool)
        l_reco_matches = np.array(self._find_matches(y_true, y_reco, mask, keep_length = True)).astype(bool)
        if mode == 'pos-match':
            all_matches = np.array(self._find_matches(y_true, y_pred, keep_length = True)).astype(bool)
            all_reco_matches = np.array(self._find_matches(y_true, y_reco, keep_length = True)).astype(bool)
            l_matches = (l_matches * np.invert(all_matches)).astype(bool)
            l_reco_matches = (l_reco_matches * np.invert(all_reco_matches)).astype(bool)

        y_pred = self.data._denormalize_targets(y_pred)
        y_true = self.data._denormalize_targets(y_true)
        y_reco = self.data._denormalize_targets(y_reco)

        if mode == 'miss':
            l_matches = (np.invert(l_matches) * y_true[:,0]).astype(bool)
            l_reco_matches = (np.invert(l_reco_matches) * y_true[:,0]).astype(bool)
            

        diff = y_true[:,:-2] - y_pred
        reco_diff = y_true[:,:-2] - y_reco

        diff = diff[l_matches]
        reco_diff = reco_diff[l_reco_matches]

        #print('{:6.0f} total Compton events'.format(np.sum(y_true[:,0])))
        #print('{:6d} NN matched events'.format(np.sum(l_matches)))
        #print('{:6d} Reco matched events'.format(np.sum(l_reco_matches)))


        fig_size = (10,4)

        def plot_hist(pos, title, width, x_min=None, x_max=None):
            plt.figure(figsize=fig_size)
            plt.title(title)
            data = diff[:,pos]

            if add_reco:
                reco_data = reco_diff[:,pos]

            if x_min is None and x_max is None:
                if add_reco:
                    x_min = min(int(np.floor(data.min())), int(np.floor(reco_data.min())))
                    x_max = max(int(np.ceil(data.max())), int(np.ceil(reco_data.max())))

                else:
                    x_min = int(np.floor(data.min()))
                    x_max = int(np.ceil(data.max()))

            x_min = np.ceil(x_min / width) * width
            x_max = ((x_max // width)+1) * width

            if add_reco:
                n, bins, _ = plt.hist(reco_data, np.arange(x_min, x_max, width), histtype='step', 
                                      label='Cut-based reco', color='tab:orange')
                #plt.plot((bins[np.argmax(n)]+bins[np.argmax(n)+1])/2, n.max(), '.', color='tab:orange')
                
            n, bins, _ = plt.hist(data, np.arange(x_min, x_max, width), histtype='step', 
                                  label='SiFi-CC NN', color='tab:blue')
            #plt.plot((bins[np.argmax(n)]+bins[np.argmax(n)+1])/2, n.max(), '.', color='tab:blue')
            
            plt.ylabel('Count')
            plt.legend()
            plt.show()

        if focus:
            plot_hist(1, 'e energy difference', .05, -2, 2)
            plot_hist(2, 'p energy difference', .05, -3, 5)
            plot_hist(3, 'e position x difference', 1.3, -20, 20)
            plot_hist(4, 'e position y difference', 1, -75, 75)
            plot_hist(5, 'e position z difference', 1.3, -20, 20)
            plot_hist(6, 'p position x difference', 1.3, -20, 20)
            plot_hist(7, 'p position y difference', 1, -75, 75)
            plot_hist(8, 'p position z difference', 1.3, -20, 20)

        else:
            plot_hist(1, 'e energy difference', .05)
            plot_hist(2, 'p energy difference', .05)
            plot_hist(3, 'e position x difference', 1.3)
            plot_hist(4, 'e position y difference', 1)
            plot_hist(5, 'e position z difference', 1.3)
            plot_hist(6, 'p position x difference', 1.3)
            plot_hist(7, 'p position y difference', 1)
            plot_hist(8, 'p position z difference', 1.3)
