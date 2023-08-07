import numpy as np
from tqdm import tqdm
from os import path
import sys

from deepfrier.utils import run_command

train_protein_ids_path = 'train_ids.npy'
test_protein_ids_path = 'test_ids.npy'
alphafold_url = 'https://alphafold.ebi.ac.uk/files/'

def download_pdbs(alphafold_dir):
    print('Loading protein ids')
    ids_path = train_protein_ids_path
    if mode == 'test':
        ids_path = test_protein_ids_path
    train_protein_ids = np.load(ids_path)
    print(train_protein_ids[0])
    af_ids = []

    for x in train_protein_ids:
        af = 'AF-'+x+'-F1-model_v4.pdb'
        af_ids.append(af)

    for file in tqdm(af_ids):
        savepath = alphafold_dir + '/' + file
        tmp_path = alphafold_dir + '/' + file + '.tmp'
        if path.exists(tmp_path):
            run_command(['rm', tmp_path])
        if not path.exists(savepath):
            url = alphafold_url + file
            run_command(['wget --quiet', url, '-O', tmp_path, 
                         '&&', 'mv', tmp_path, savepath])

download_dir = sys.argv[1]
t5_embeds_dir = sys.argv[2]
mode = sys.argv[3]
train_protein_ids_path = path.join(t5_embeds_dir, train_protein_ids_path)
test_protein_ids_path = path.join(t5_embeds_dir, test_protein_ids_path)

download_pdbs(download_dir, mode)