## Imports

import PyPDF2
from numpy import zeros
from os import listdir
from os.path import join
from tqdm import tqdm
from math import exp, log
from random import shuffle, uniform, sample

## Methods

def clean_text(text, chars, header):
    '''Filters an input text to only contain lowercase letters and the space 
    character.'''
    text = text.lower()
    if header:
        text = ' '.join(text.split(':')[1:])
    text = text.replace('\n', ' ')
    text = text.replace(' .', '.')
    cleaned = filter(chars.__contains__, text)
    return ''.join(cleaned) + ' '

def get_text(path, char_dict, header=False):
    '''Reads in and cleans the text from a .pdf or .txt file.'''
    text_type = path[-3:]
    if text_type == 'pdf':
        doc = PyPDF2.PdfReader(path)
        text = ''
        for page in doc.pages:
            text += clean_text(page.extract_text(), list(char_dict.keys()), header)
    elif text_type == 'txt':
        with open(path, 'r') as file:
            text = clean_text(file.read(), list(char_dict.keys()), header)
    else:
        raise RuntimeError('Import file must be of type .pdf or .txt!')
    return text

def get_q(text, char_dict):
    '''Builds Q and P from a given text and char_dict.'''
    char_len = len(list(char_dict.keys()))
    q = zeros((char_len, char_len))
    p = zeros((char_len))
    p[char_dict[text[0]]] += 1
    for i in range(1, len(text)):
        q[char_dict[text[i - 1]]][char_dict[text[i]]] += 1
        p[char_dict[text[i]]] += 1
    for i in range(len(q)):
        for j in range(len(q[i])):
            q[i][j] = max(q[i][j], 1)
        q[i] = q[i] / sum(q[i])
    return q, p / len(text)

def new_perm(permutation, permutations):
    '''Computes a new random permutation'''
    for _ in range(permutations):
        ab = sample(range(0, len(permutation) - 1), 2)
        a, b = ab[0], ab[1]
        permutation = list(permutation)
        permutation[a], permutation[b] = permutation[b], permutation[a]
        new_permutation = ''.join(permutation)
    return new_permutation

def transition(sigma, char_dict, encoded, display_amount=None):
    '''Computes transition on a given text.'''
    data = ''
    display_amount = len(encoded) if display_amount == None else display_amount
    for i in range(display_amount):
        data += sigma[char_dict[encoded[i]]]
    return data

def energy_func(sigma, char_dict, encoded, q, p):
    '''Computes the energy on a permuted text.'''
    trans = transition(sigma, char_dict, encoded)
    likelihood = log(p[char_dict[trans[0]]])
    for j in range(1, len(encoded)):
        likelihood -= log(q[char_dict[trans[j-1]]][char_dict[trans[j]]])
    return likelihood

def main(specific_text=None, verbose=True, save=True):
    '''Runs MCMC using all files in /text_data on all files in /encoded_text
    and outputs results in /decoded_text.'''
    char_dict = {x: i for i, x in enumerate(' abcdefghijklmnopqrstuvwxyz')}
    chars = list(char_dict.keys())
    shuffle(chars)
    permutation = ''.join(chars)

    # hyper parameters
    beta = 0.63 # tunable hyperparameter (best for all = 0.63)
    permutations = 2 # number of times text is permuted before scoring
    convergence_delta = 2000 # max number of worse iterations before stopping
    max_epochs = 25000 # maximum number of iterations to run MCMC


    X0 = ''
    text_dir = './text_data'
    decode_dir = './decoded_text'
    print('Building q and p...')
    for filename in listdir(text_dir):
        try:
            X0 += get_text(join(text_dir, filename), char_dict)
        except RuntimeError as e:
            None

    q, p = get_q(X0, char_dict)

    print('Running mcmc on encoded texts...')
    encoded_dir = './encoded_text'
    for filename in tqdm(listdir(encoded_dir), desc='File:'):
        if specific_text != None and filename != specific_text:
            continue
        header = filename.split('.')[0]
        try:
            encoded = get_text(join(encoded_dir, filename), char_dict, True)
            convergence_counter = 0
            for i in range(max_epochs):
                curr = new_perm(permutation, permutations)
                e_curr = energy_func(curr, char_dict, encoded, q, p)
                e_prev = energy_func(permutation, char_dict, encoded, q, p)
                e_delta = e_curr - e_prev
                if e_delta < 0 or uniform(0, 1) < exp((-beta) * e_delta):
                    permutation = curr
                    if verbose:
                        print(f'{i}: ' + transition(permutation, char_dict, encoded, 80))
                    convergence_counter = 0
                else:
                    convergence_counter += 1
                    if convergence_counter >= convergence_delta:
                        break
            print(f'Permutation: {permutation}')
            print('Decoded text: \n')
            print(transition(permutation, char_dict, encoded, 80))
            if save:
                with open(join(decode_dir, f'{header}_decoded.txt'), 'w') as f:
                    f.write(transition(permutation, char_dict, encoded))
                print(f'Saved {header}_decoded.txt')
        except RuntimeError as e:
            None
        print('Enter any key to continue: ')
        _ = input()

## Run

save_data = False
# main(specific_text='student_20_text1.txt', save=save_data)
# main(specific_text='student_219_text2.txt', save=save_data)
# main(specific_text='student_102_text3.txt', save=save_data)
main(save=save_data)