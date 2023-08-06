import sys
import glob
import numpy as np
from os import path
from tqdm import tqdm

def read_okayed_chunks(datasets_dir, aspect):
    okay_chunks = glob.glob(datasets_dir + '/' + aspect + '/*.okay')
    chunk_n = [path.basename(x).split('.')[0].split('_')[-1] for x in okay_chunks]
    chunk_n.sort()
    feature_chunks = [datasets_dir + '/' + aspect + '/' + aspect + '_structure_features_' + n + '.npy'
                      for n in chunk_n]
    name_chunks = [datasets_dir + '/' + aspect + '/' + aspect + '_structure_names_' + n + '.txt'
                      for n in chunk_n]
    for index in range(len(chunk_n)):
        if path.exists(feature_chunks[index]) and path.exists(name_chunks[index]):
            pass
        else:
            print('Does not exist')
            print(feature_chunks[index], name_chunks[index])
            quit()

    assert len(feature_chunks) == len(name_chunks)
    return feature_chunks, name_chunks

def concat_all(datasets_dir, name_chunks, feature_chunks, aspect):
    output_names_file = datasets_dir + '/' + aspect + '_structure_names.txt'
    output_features_file = datasets_dir + '/' + aspect + '_structure_features.npy'
    names_vec = []
    features_vec = np.array([])
    for index in tqdm(range(len(name_chunks))):
        name_chunk_path = name_chunks[index]
        feature_chunk_path = feature_chunks[index]
        lines = open(name_chunk_path, 'r').read().split('\n')
        lines = [x for x in lines if len(x) > 0]

        feature_chunk = np.load(feature_chunk_path)
        #assert len(lines) == len(feature_chunk)
        if len(features_vec) > 0:
            #print(feature_chunk[0])
            features_vec = np.concatenate((features_vec, feature_chunk))
        else:
            features_vec = feature_chunk

        names_vec += lines

    np.save(output_features_file, features_vec)
    with open(output_names_file, 'w') as output:
        for line in names_vec:
            output.write(line.split('-')[1] +'\n')

download_dir = sys.argv[1]

aspects = ['mf', 'bp', 'cc']

for aspect in aspects:
    feature_chunks, name_chunks = read_okayed_chunks(download_dir, aspect)
    print(aspect, len(feature_chunks))
    concat_all(download_dir, name_chunks, feature_chunks, aspect)