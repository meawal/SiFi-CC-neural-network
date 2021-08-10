import numpy as np
from sificc_lib import utils

class DataModelQlty():
    '''Data model for the features and targets to train SiFi-CC Quality 
    Neural Network. The training data should be generated seperately 
    from a trained SiFi-CC Neural Network.
    
    Features R_n*(9*clusters_limit) format: {
        cluster entries, 
        cluster energy, 
        cluster energy uncertainty, 
        cluster position (x,y,z), 
        cluster position uncertainty (x,y,z) 
    } * clusters_limit
        
    Targets R_n*11 format: {
        event type (is ideal Compton or not),
        e energy,
        p energy,
        e position (x,y,z),
        p position (x,y,z),
        e cluster index,
        p cluster index,
    }
    
    Reco R_n*9 format: {
        event type (is ideal Compton or not),
        e energy,
        p energy,
        e position (x,y,z),
        p position (x,y,z),
    }
    
    Quality R_n*4 format: {
        e energy quality,
        p energy quality,
        e position quality,
        p position quality,
    }
    '''
    def __init__(self, file_name, *, batch_size = 64, validation_percent = .05, test_percent = .1, 
                 weight_compton = 1, weight_non_compton = .75):
        self.validation_percent = validation_percent
        self.test_percent = test_percent
        self.batch_size = batch_size
        self.weight_compton = weight_compton
        self.weight_non_compton = weight_non_compton
        
        self.cluster_size = 9
        self.append_dim = True
        
        self.__std_factor = 10
        
        # loading training matrices
        with open(file_name, 'rb') as f_train:
            npz = np.load(f_train)
            self._features = npz['features']
            self._targets = npz['targets']
            self._reco = npz['reco']
            self._seq = npz['sequence']
            self._qlty = npz['quality']
            
        # assert number of columns is correct
        assert self._features.shape[1] % self.cluster_size == 0
        
        # define clusters limit
        self.clusters_limit = self._features.shape[1] // self.cluster_size
        
        # define number of events
        self.length = self._targets.shape[0]
        
        #normalize features, targets, and reco
        self._features = (self._features - self.__mean_features) / self.__std_features
        self._targets = (self._targets - self.__mean_targets) / self.__std_targets
        self._reco = (self._reco - self.__mean_targets[:-2]) / self.__std_targets[:-2]
        
    def _denormalize_features(self, data):
        if data.shape[-1] == self._features.shape[-1]:
            return (data * self.__std_features) + self.__mean_features
        raise Exception('data has invalid shape of {}'.format(data.shape))
    
    def _denormalize_targets(self, data):
        if data.shape[-1] == self._targets.shape[-1]:
            return (data * self.__std_targets) + self.__mean_targets
        elif data.shape[-1] == self._reco.shape[-1]:
            return (data * self.__std_targets[:-2]) + self.__mean_targets[:-2]
        else:
            raise Exception('data has invalid shape of {}'.format(data.shape))
    
    def normalize_targets(self, data):
        if data.shape[-1] == self._targets.shape[-1]:
            return (data - self.__mean_targets) / self.__std_targets
        elif data.shape[-1] == self._reco.shape[-1]:
            return (data - self.__mean_targets[:-2]) / self.__std_targets[:-2]
        else:
            raise Exception('data has invalid shape of {}'.format(data.shape))
    
    def get_targets_dic(self, start=None, end=None):
        start = start if start is not None else 0
        end = end if end is not None else self.length
        
        return {
            'type': self._target_type[start:end],
            'e_cluster': self._target_e_cluster[start:end],
            'p_cluster': self._target_p_cluster[start:end],
            'pos_x': self._target_pos_x[start:end],
            'pos_y': self._target_pos_y[start:end],
            'pos_z': self._target_pos_z[start:end],
            'energy': self._target_energy[start:end],
            'quality': self._target_qlty[start:end]
        }
    
    def get_features(self, start=None, end=None):
        start = start if start is not None else 0
        end = end if end is not None else self.length
        
        if self.append_dim:
            return self._features[start:end].reshape((-1, self._features.shape[1], 1))
        else:
            return self._features[start:end]
        
    def shuffle(self, only_train=True):
        limit = self.validation_start_pos if only_train else self.length
        sequence = np.arange(self.length)
        sequence[:limit] = np.random.permutation(limit)
        
        self._features = self._features[sequence]
        self._targets = self._targets[sequence]
        self._reco = self._reco[sequence]
        self._seq = self._seq[sequence]
        self._qlty = self._qlty[sequence]
        
    @property
    def steps_per_epoch(self):
        return int(np.ceil(self.validation_start_pos/self.batch_size))
    
    def generate_batch(self, shuffle=True, augment=False):
        while True:
            if shuffle:
                self.shuffle(only_train=True)

            for step in range(self.steps_per_epoch):
                start = step * self.batch_size
                end = (step+1) * self.batch_size
                # end should not enter the validation range
                end = end if end <= self.validation_start_pos else self.validation_start_pos
                
                features_batch = self.get_features(start, end)
                targets_batch = self.get_targets_dic(start, end)
                
                if augment:
                    sequence, expanded_sequence = self.__get_augmentation_sequence()
                    features_batch = features_batch[:,expanded_sequence]
                    targets_batch['e_cluster'][:,1] = np.where(np.equal(targets_batch['e_cluster'][:,[1]], sequence))[1]
                    targets_batch['p_cluster'][:,1] = np.where(np.equal(targets_batch['p_cluster'][:,[1]], sequence))[1]
                
                yield (
                    features_batch, 
                    targets_batch, 
                    targets_batch['type'] * self.weight_compton + \
                        (1-targets_batch['type']) * self.weight_non_compton
                )
        
    def __get_augmentation_sequence(self):
        num_clusters = self.clusters_limit
        sequence = np.random.permutation(num_clusters)
        expanded_sequence = np.repeat(sequence * self.cluster_size, self.cluster_size) + \
                            np.tile(np.arange(self.cluster_size), num_clusters)
        return sequence, expanded_sequence
    
    def shuffle_training_clusters(self):
        # e_pos = 9
        # p_pos = 10
        for i in range(self.length):
            sequence, expanded_sequence = self.__get_augmentation_sequence()
            self._features[i] = self._features[i, expanded_sequence]
            self._targets[i,9] = np.where(np.equal(self._targets[i,9], sequence))[0]
            self._targets[i,10] = np.where(np.equal(self._targets[i,10], sequence))[0]
    
    ################# Properties #################
    @property
    def validation_start_pos(self):
        return int(self.length * (1-self.validation_percent-self.test_percent))
    
    @property
    def test_start_pos(self):
        return int(self.length * (1-self.test_percent))
    
    @property
    def train_x(self):
        return self.get_features(None, self.validation_start_pos)
    
    @property
    def train_y(self):
        return self.get_targets_dic(None, self.validation_start_pos)
    
    @property
    def train_row_y(self):
        return self._targets[:self.validation_start_pos]
    
    @property
    def validation_x(self):
        return self.get_features(self.validation_start_pos, self.test_start_pos)
    
    @property
    def validation_y(self):
        return self.get_targets_dic(self.validation_start_pos, self.test_start_pos)
    
    @property
    def validation_row_y(self):
        return self._targets[self.validation_start_pos: self.test_start_pos]
    
    @property
    def test_x(self):
        return self.get_features(self.test_start_pos, None)
    
    @property
    def test_y(self):
        return self.get_targets_dic(self.test_start_pos, None)
    
    @property
    def test_row_y(self):
        return self._targets[self.test_start_pos:]
    
    @property
    def reco_valid(self):
        return self._reco[self.validation_start_pos: self.test_start_pos]
    
    @property
    def reco_test(self):
        return self._reco[self.test_start_pos:]
    
    @property
    def __mean_features(self):
        # define normalization factors
        mean_entries = [1.780826610712862]
        mean_energies = [1.3134160095873435]
        mean_energies_unc = [0.03338954916639435]
        mean_positions = [ 3.08482929e+02,  4.42330610e-02, -8.50224908e-01]
        mean_positions_unc = [ 1.0542618,  12.85909077,  0.94718083]
        
        # declare the mean of a single cluster and repeat it throughout the clusters
        mean = np.concatenate((
            mean_entries, 
            mean_energies, 
            mean_energies_unc, 
            mean_positions, 
            mean_positions_unc
        ))
        mean = np.tile(mean, self.clusters_limit)
        return mean
    
    @property
    def __std_features(self):
        # define normalization factors
        std_entries = [1.6371731651624135]
        std_energies = [1.874215149152707]
        std_energies_unc = [0.0250835375536536]
        std_positions = [97.33908375, 28.98121222, 27.53657139]
        std_positions_unc = [1.00549903, 6.10554312, 0.75687835]
        
        std = np.concatenate((
            std_entries, 
            std_energies, 
            std_energies_unc, 
            std_positions, 
            std_positions_unc
        ))
        std = np.tile(std, self.clusters_limit)
        return std
    
    @property
    def __mean_targets(self):
        mean_e_energy = [1.145902968123442]
        mean_p_energy = [2.22115921342183]
        mean_e_position = [2.03054229e+02, -1.05712158e-01, -3.13056242e+00]
        mean_p_position = [3.92947326e+02, 5.97191421e-02, 1.18562119e+00]
        
        mean = np.concatenate((
            [0],
            mean_e_energy, 
            mean_p_energy,
            mean_e_position,
            mean_p_position,
            [0,0]
        ))
        return mean
    
    @property
    def __std_targets(self):
        std_e_energy = [1.7225469452392403] / np.array(self.__std_factor)
        std_p_energy = [1.7916010745717312] / np.array(self.__std_factor)
        std_e_position = [23.54037899, 20.70622677, 27.26802304] / np.array(self.__std_factor)
        std_p_position = [26.51553238, 28.33107502, 28.13506062] / np.array(self.__std_factor)

        std = np.concatenate((
            [1],
            std_e_energy, 
            std_p_energy,
            std_e_position,
            std_p_position,
            [1,1]
        ))
        return std
    
    
    @property
    def _target_type(self):
        # [t]
        return self._targets[:,[0]]
    
    @property
    def _target_e_cluster(self):
        # [t, e_clus]
        return self._targets[:,[0,9]]
    
    @property
    def _target_p_cluster(self):
        # [t, p_clus]
        return self._targets[:,[0,10]]
    
    @property
    def _target_pos_x(self):
        # [t, e_clus, e_pos_x, p_clus, p_pos_x]
        return self._targets[:,[0,9,3,10,6]]
    
    @property
    def _target_pos_y(self):
        # [t, e_clus, e_pos_y, p_clus, p_pos_y]
        return self._targets[:,[0,9,4,10,7]]
    
    @property
    def _target_pos_z(self):
        # [t, e_clus, e_pos_z, p_clus, p_pos_z]
        return self._targets[:,[0,9,5,10,8]]
    
    @property
    def _target_energy(self):
        # [t, e_enrg, p_enrg]
        return self._targets[:,[0,1,2]]
    
    @property
    def _target_qlty(self):
        # [t, e_energy_qlty, p_energy_qlty, e_pos_qlty, p_pos_qlty]
        return np.concatenate([self._targets[:,[0]], self._qlty], axis=1)
    