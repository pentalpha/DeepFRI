import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

import glob
import json
import argparse
from multiprocessing import Pool
from os import path, mkdir
from deepfrier.Predictor import Predictor
from deepfrier.utils import chunks

def read_pdbdir(pdbdir_path):
    pdb_fn_list = glob.glob(pdbdir_path + '/*.pdb*')
    pdb_fn_list = [x for x in pdb_fn_list if not x.endswith('.tmp')]
    pdb_fn_list.sort()
    return pdb_fn_list

def predict_features(pdb_sublist, sublist_name, subdataset_dir, model, gcn, ont):
    okay_path = path.join(subdataset_dir, 
            ont+'_'+sublist_name+'.okay')
    if not path.exists(okay_path):
        predictor = Predictor(model, gcn=gcn, ont=ont)
        predictor.predict_from_PDB_list(pdb_sublist, sublist_name, subdataset_dir)
        open(okay_path, 'w').write('okay')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str,  help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str,  help="Protein contact map to be annotated (in *npz file format).")
    parser.add_argument('-pdb', '--pdb_fn', type=str,  help="Protein PDB file to be annotated.")
    parser.add_argument('--cmap_csv', type=str,  help="Catalogue with chain to file path mapping.")
    parser.add_argument('--pdb_dir', type=str,  help="Directory with PDB files of predicted Rosetta/DMPFold structures.")
    parser.add_argument('--fasta_fn', type=str,  help="Fasta file with protein sequences.")
    parser.add_argument('--model_config', type=str, default='./trained_models/model_config.json', help="JSON file with model names.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', required=True, choices=['mf', 'bp', 'cc', 'ec'],
                        help="Gene Ontology/Enzyme Commission.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI', help="Save predictions/saliency in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--use_guided_grads', help="Use guided grads to compute gradCAM.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein and every MF-GO term/EC number.", action="store_true")
    args = parser.parse_args()

    with open(args.model_config) as json_file:
        params = json.load(json_file)

    if args.seq is not None or args.fasta_fn is not None:
        params = params['cnn']
    elif args.cmap is not None or args.pdb_fn is not None or args.cmap_csv is not None or args.pdb_dir is not None:
        params = params['gcn']
    gcn = params['gcn']
    layer_name = params['layer_name']
    models = params['models']

    processes = 3
    pdbs = read_pdbdir(args.pdb_dir)
    pdb_chunks = []
    i = 0
    for chunk in chunks(pdbs, 200):
        pdb_chunks.append(chunk)
        i += 1
        #if i == processes*2:
        #    break
    
    #print(pdb_chunks)

    dataset_path = 'dataset'
    if not path.exists(dataset_path):
        mkdir(dataset_path)
    
    for ont in ['mf', 'bp', 'cc']:
        with Pool(3) as p:
            subdataset_dir = path.join(dataset_path, ont)
            if not path.exists(subdataset_dir):
                mkdir(subdataset_dir)
            #predictor = lambda list_tp: predict_features(list_tp[0], list_tp[1], subdataset_dir, models[ont], gcn, ont)
            arg_list = []
            for chunk_i in range(len(pdb_chunks)):
                arg_list.append((pdb_chunks[chunk_i], str(chunk_i), subdataset_dir, models[ont], gcn, ont))
            #print(arg_list)
            p.starmap(predict_features, arg_list)
