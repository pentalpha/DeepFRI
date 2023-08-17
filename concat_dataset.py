import sys
import glob
import numpy as np
from os import path, mkdir
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
     
    return output_names_file, output_features_file

dataset_dir = sys.argv[1]
t5_embeds_dir = sys.argv[2]
mode = sys.argv[3]
protein_ids_path = path.join(t5_embeds_dir, mode+'_ids.npy')
embeds_path = path.join(t5_embeds_dir, mode+'_embeds.npy')

aspects = ['mf', 'bp', 'cc']
aspect_dfs = []
for aspect in aspects:
    feature_chunks, name_chunks = read_okayed_chunks(dataset_dir, aspect)
    print(aspect, len(feature_chunks))
    output_names_file, output_features_file = concat_all(dataset_dir, name_chunks, feature_chunks, aspect)
    aspect_dfs.append((aspect, output_names_file, output_features_file))


final_dataset_path = 'final_dataset'
if not path.exists(final_dataset_path):
    mkdir(final_dataset_path)

print('Loading T5EMBEDS')
t5_names  = np.load(protein_ids_path)
t5_embeds = np.load(embeds_path)

print('Indexing T5EMBEDS')
prot_name_to_index = {t5_names[i]: i for i in range(len(t5_names))}

print('Adding sequential features to structural features')
for aspect, names_file, features_file in aspect_dfs:
    print(aspect, names_file, features_file)
    deepfri_names = open(names_file, 'r').read().rstrip('\n').split('\n')
    deepfri_features = np.load(features_file)
    print('Joining features')
    all_embeds = [np.concatenate([t5_embeds[prot_name_to_index[deepfri_names[i]]], 
               deepfri_features[i]]) for i in tqdm(range(len(deepfri_names)))]
    print('Converting to np array')
    all_embeds = np.asarray(all_embeds)
    features_final_path = path.join(final_dataset_path, mode+'_features_'+aspect+'.npy')
    names_final_path = path.join(final_dataset_path, mode+'_ids_'+aspect+'.npy')
    print('Saving')
    np.save(features_final_path, all_embeds)
    np.save(names_final_path, np.array(deepfri_names))
