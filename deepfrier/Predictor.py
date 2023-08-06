import os
from os import path
import csv
import glob
import json
import gzip
import secrets

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .utils import load_catalogue, load_FASTA, load_predicted_PDB, seq2onehot
from .layers import MultiGraphConv, GraphConv, FuncPredictor, SumPooling


class GradCAM(object):
    """
    GradCAM for protein sequences.
    [Adjusted for GCNs based on https://arxiv.org/abs/1610.02391]
    """
    def __init__(self, model, layer_name="GCNN_concatenate"):
        self.grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    def _get_gradients_and_filters(self, inputs, class_idx, use_guided_grads=False):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(inputs)
            loss = predictions[:, class_idx, 0]
        grads = tape.gradient(loss, conv_outputs)

        if use_guided_grads:
            grads = tf.cast(conv_outputs > 0, "float32")*tf.cast(grads > 0, "float32")*grads

        return conv_outputs, grads

    def _compute_cam(self, output, grad):
        weights = tf.reduce_mean(grad, axis=1)
        # perform weighted sum
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1).numpy()

        return cam

    def heatmap(self, inputs, class_idx, use_guided_grads=False):
        output, grad = self._get_gradients_and_filters(inputs, class_idx, use_guided_grads=use_guided_grads)
        cam = self._compute_cam(output, grad)
        heatmap = (cam - cam.min())/(cam.max() - cam.min())

        return heatmap.reshape(-1)


class Predictor(object):
    """
    Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """
    def __init__(self, model_prefix, gcn=True, ont=None):
        self.ont_code = ont
        self.model_prefix = model_prefix
        self.gcn = gcn
        self._load_model()

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_prefix + '.hdf5',
                                                custom_objects={'MultiGraphConv': MultiGraphConv,
                                                                'GraphConv': GraphConv,
                                                                'FuncPredictor': FuncPredictor,
                                                                'SumPooling': SumPooling})
        # load parameters
        with open(self.model_prefix + "_model_params.json") as json_file:
            metadata = json.load(json_file)

        self.gonames = np.asarray(metadata['gonames'])
        self.goterms = np.asarray(metadata['goterms'])
        self.thresh = 0.1*np.ones(len(self.goterms))
        
        print('\nConcat layer:')
        self.concat_layer = self.model.get_layer(index=11)
        print(self.concat_layer)
        print(type(self.concat_layer))
        self.intermediate_layer_model = tf.keras.Model(inputs=self.model.input, outputs=self.concat_layer.output)

    def _load_cmap(self, filename, cmap_thresh=10.0):
        if filename.endswith('.pdb'):
            D, seq = load_predicted_PDB(filename)
            A = np.double(D < cmap_thresh)
        elif filename.endswith('.npz'):
            cmap = np.load(filename)
            if 'C_alpha' not in cmap:
                raise ValueError("C_alpha not in *.npz dict.")
            D = cmap['C_alpha']
            A = np.double(D < cmap_thresh)
            seq = str(cmap['seqres'])
        elif filename.endswith('.pdb.gz'):
            rnd_fn = "".join([secrets.token_hex(10), '.pdb'])
            with gzip.open(filename, 'rb') as f, open(rnd_fn, 'w') as out:
                out.write(f.read().decode())
            D, seq = load_predicted_PDB(rnd_fn)
            A = np.double(D < cmap_thresh)
            os.remove(rnd_fn)
        else:
            raise ValueError("File must be given in *.npz or *.pdb format, not:"+filename)
        # ##
        S = seq2onehot(seq)
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)

        return A, S, seq

    def predict(self, test_prot, cmap_thresh=10.0, chain='query_prot'):
        print ("### Computing predictions on a single protein...")
        self.Y_hat = np.zeros((1, len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.test_prot_list = [chain]
        if self.gcn:
            print('self.gcn')
            A, S, seqres = self._load_cmap(test_prot, cmap_thresh=cmap_thresh)

            y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]

            '''print('Hidden output:')
            hidden_output = self.intermediate_layer_model([A, S], training=False).numpy().reshape(-1)
            print(hidden_output_raw.shape)
            hidden_output = hidden_output_raw[:, :, 0]
            print(hidden_output.shape)
            #print(hidden_output[0])
            print(hidden_output.reshape(-1).shape)
            #print(hidden_output.reshape(-1))'''
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
        else:
            print('seq2onehot')
            S = seq2onehot(str(test_prot))
            S = S.reshape(1, *S.shape)
            print(S[0])
            print(S.shape)
            y = self.model(S, training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], test_prot]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_PDB_list(self, pdb_fn_list, sublist_name, subdataset_dir, cmap_thresh=10.0):
        self.chain2path = {pdb_fn.split('/')[-1].split('.')[0]: pdb_fn for pdb_fn in pdb_fn_list}
        self.test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        
        pdb_names = []
        input_data = []
        for i, chain in enumerate(self.test_prot_list):
            pdb_name = self.chain2path[chain].split('/')[-1]
            try:
                A, S, seqres = self._load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
                input_data.append((pdb_name, A, S))
                pdb_names.append(pdb_name)
            except Exception as err:
                print('Error loading', self.chain2path[chain], pdb_name)
                print(err)
        
        structure_features = []
        n = 0
        bar = tqdm(total=len(input_data))
        for pdb_name, A, S in input_data:
            hidden_output = self.intermediate_layer_model([A, S], training=False).numpy().reshape(-1)
            #print(hidden_output.shape)
            structure_features.append(hidden_output)
            n += 1
            bar.update(1)
        bar.close()
        print('Saving names')
        structure_names_path = path.join(subdataset_dir, 
            self.ont_code+'_structure_names_'+sublist_name+'.txt')
        open(structure_names_path, 'w').write("\n".join(pdb_names)+'\n')
        print("Converting features to numpy df")
        structure_features = np.asarray(structure_features)
        print('Saving features')
        structure_features_path = path.join(subdataset_dir, 
            self.ont_code+'_structure_features_'+sublist_name)
        np.save(structure_features_path, structure_features)

    def predict_from_PDB_dir(self, dir_name, cmap_thresh=10.0):
        print ("### Computing predictions from directory with PDB files...")
        pdb_fn_list = glob.glob(dir_name + '/*.pdb*')
        pdb_fn_list = [x for x in pdb_fn_list if not x.endswith('.tmp')]
        pdb_fn_list.sort()
        self.chain2path = {pdb_fn.split('/')[-1].split('.')[0]: pdb_fn for pdb_fn in pdb_fn_list}
        self.test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        total = len(self.test_prot_list)
        bar = tqdm(total=3000)
        structure_features = []
        pdb_names = []
        n = 0
        for i, chain in enumerate(self.test_prot_list):
            pdb_name = self.chain2path[chain].split('/')[-1]
            pdb_names.append(pdb_name)
            A, S, seqres = self._load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
            hidden_output = self.intermediate_layer_model([A, S], training=False).numpy().reshape(-1)
            #print(hidden_output.shape)
            structure_features.append(hidden_output)
            n += 1
            if n >= 3000:
                break
            bar.update(1)
            '''y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))'''
        bar.close()
        print('Saving names')
        open(self.ont_code+'_structure_names.txt', 'w').write("\n".join(pdb_names)+'\n')
        print("Converting features to numpy df")
        structure_features = np.asarray(structure_features)
        print('Saving features')
        np.save(self.ont_code+"_structure_features.npy", structure_features)


    def predict_from_catalogue(self, catalogue_fn, cmap_thresh=10.0):
        print ("### Computing predictions from catalogue...")
        self.chain2path = load_catalogue(catalogue_fn)
        self.test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        for i, chain in enumerate(self.test_prot_list):
            A, S, seqres = self._load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
            y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_fasta(self, fasta_fn):
        print ("### Computing predictions from fasta...")
        self.test_prot_list, sequences = load_FASTA(fasta_fn)
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}

        for i, chain in enumerate(self.test_prot_list):
            S = seq2onehot(str(sequences[i]))
            S = S.reshape(1, *S.shape)
            y = self.model(S, training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], str(sequences[i])]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def save_predictions(self, output_fn):
        print ("### Saving predictions to *.json file...")
        # pickle.dump({'pdb_chains': self.test_prot_list, 'Y_hat': self.Y_hat, 'goterms': self.goterms, 'gonames': self.gonames}, open(output_fn, 'wb'))
        with open(output_fn, 'w') as fw:
            out_data = {'pdb_chains': self.test_prot_list,
                        'Y_hat': self.Y_hat.tolist(),
                        'goterms': self.goterms.tolist(),
                        'gonames': self.gonames.tolist()}
            json.dump(out_data, fw, indent=1)

    def export_csv(self, output_fn, verbose):
        with open(output_fn, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')
            writer.writerow(['### Predictions made by DeepFRI.'])
            writer.writerow(['Protein', 'GO_term/EC_number', 'Score', 'GO_term/EC_number name'])
            if verbose:
                print ('Protein', 'GO-term/EC-number', 'Score', 'GO-term/EC-number name')
            for prot in self.prot2goterms:
                sorted_rows = sorted(self.prot2goterms[prot], key=lambda x: x[2], reverse=True)
                for row in sorted_rows:
                    if verbose:
                        print (prot, row[0], '{:.5f}'.format(row[2]), row[1])
                    writer.writerow([prot, row[0], '{:.5f}'.format(row[2]), row[1]])
        csvFile.close()

    def compute_GradCAM(self, layer_name='GCNN_concatenate', use_guided_grads=False):
        print ("### Computing GradCAM for each function of every predicted protein...")
        gradcam = GradCAM(self.model, layer_name=layer_name)

        self.pdb2cam = {}
        for go_indx in self.goidx2chains:
            pred_chains = list(self.goidx2chains[go_indx])
            print ("### Computing gradCAM for ", self.gonames[go_indx], '... [# proteins=', len(pred_chains), ']')
            for chain in pred_chains:
                if chain not in self.pdb2cam:
                    self.pdb2cam[chain] = {}
                    self.pdb2cam[chain]['GO_ids'] = []
                    self.pdb2cam[chain]['GO_names'] = []
                    self.pdb2cam[chain]['sequence'] = None
                    self.pdb2cam[chain]['saliency_maps'] = []
                self.pdb2cam[chain]['GO_ids'].append(self.goterms[go_indx])
                self.pdb2cam[chain]['GO_names'].append(self.gonames[go_indx])
                self.pdb2cam[chain]['sequence'] = self.data[chain][1]
                self.pdb2cam[chain]['saliency_maps'].append(gradcam.heatmap(self.data[chain][0], go_indx, use_guided_grads=use_guided_grads).tolist())

    def save_GradCAM(self, output_fn):
        print ("### Saving CAMs to *.json file...")
        # pickle.dump(self.pdb2cam, open(output_fn, 'wb'))
        with open(output_fn, 'w') as fw:
            json.dump(self.pdb2cam, fw, indent=1)
