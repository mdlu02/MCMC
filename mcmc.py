## Imports

import PyPDF2
from numpy import zeros
from os import listdir
from os.path import join
from tqdm import tqdm
from math import exp, log
from random import shuffle, uniform, sample, randint

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
            text += clean_text(page.extract_text(), \
                               list(char_dict.keys()), header)
    elif text_type == 'txt':
        with open(path, 'r') as file:
            text = clean_text(file.read(), list(char_dict.keys()), header)
    else:
        raise RuntimeError('Import file must be of type .pdf or .txt!')
    return text

def get_q(text, char_dict):
    '''Builds Q and P from a given text and char_dict using the digram model.'''
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

def new_perm(perm, permutations, visited, var=0):
    '''Computes a new random permutation. Allows for some random variance in the
    number of changes made for the new permutations.'''
    changes = permutations
    if var != 0:
        rand = randint(0, 100)
        if rand >= 90:
            changes = permutations + var
        elif rand <= 10:
            changes = permutations - var
        else:
            changes = permutations
    while True:
        curr = perm
        for _ in range(changes):
            ab = sample(range(0, len(curr) - 1), 2)
            a, b = ab[0], ab[1]
            curr = list(curr)
            curr[a], curr[b] = curr[b], curr[a]
            new_permutation = ''.join(curr)
        if new_permutation not in visited:
            break
    return new_permutation

def transition(perm, char_dict, encoded, display_amount=None):
    '''Computes transition on a given text.'''
    data = ''
    display_amount = len(encoded) if display_amount == None else display_amount
    for i in range(display_amount):
        data += perm[char_dict[encoded[i]]]
    return data

def energy_func(perm1, perm2, char_dict, encoded, q, p, lim=None):
    '''Computes the energy delta on a permuted texts.'''
    trans1 = transition(perm1, char_dict, encoded)
    trans2 = transition(perm2, char_dict, encoded)
    if lim == None:
        lim = len(trans1)
    delta = log(p[char_dict[trans1[0]]]) - log(p[char_dict[trans2[0]]])
    for j in range(1, lim):
        delta -= log(q[char_dict[trans1[j-1]]][char_dict[trans1[j]]]) - \
                        log(q[char_dict[trans2[j-1]]][char_dict[trans2[j]]])
    return delta

def main(specific_text=None, verbose=True, save=True):
    '''Runs MCMC using all files in /text_data on all files in /encoded_text
    and outputs results in /decoded_text.'''
    char_dict = {x: i for i, x in enumerate(' abcdefghijklmnopqrstuvwxyz')}
    chars = list(char_dict.keys())
    shuffle(chars)
    perm = ''.join(chars)

    # hyper parameters
    beta = 0.635 # tunable hyperparameter (best for all = 0.635)
    permutations = 2 # number of times text is permuted before scoring
    var = 0 # Value to shift permutations under a random conditioin
    lim = None # First x characters to consider for energy calculation
    convergence_delta = 2000 # max number of worse iterations before stopping
    max_epochs = 10000 # maximum number of iterations to run MCMC


    X0 = ''
    text_dir = './text_data'
    decode_dir = './decoded_text'
    print('Building q and p...')
    for filename in tqdm(listdir(text_dir)):
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
        print(f'File: {filename}...')
        try:
            encoded = get_text(join(encoded_dir, filename), char_dict, True)
            convergence_counter = 0
            visited = set()
            for i in range(max_epochs):
                curr = new_perm(perm, permutations, visited, var=var)
                visited.add(curr)
                e_delta = energy_func(curr, perm, char_dict, encoded, q, p, lim)
                if e_delta < 0 or uniform(0, 1) < exp((-beta) * e_delta):
                    perm = curr
                    visited.clear()
                    if verbose:
                        print(f'{i}: ' + \
                                transition(perm, char_dict, encoded, 80))
                    convergence_counter = 0
                else:
                    convergence_counter += 1
                    if convergence_counter >= convergence_delta:
                        break
            print(f'Permutation: {perm}')
            print('Decoded text: \n')
            print(transition(perm, char_dict, encoded, 80))
            print('Enter any key to continue: ')
            _ = input()
            if save:
                with open(join(decode_dir, f'{header}_decoded.txt'), 'w') as f:
                    f.write(transition(perm, char_dict, encoded))
                print(f'Saved {header}_decoded.txt')
        except RuntimeError as e:
            None

## Run

save_data = False
# main(specific_text='student_219_text2.txt', save=save_data)
main(specific_text='student_20_text1.txt', save=save_data)
# main(specific_text='student_102_text3.txt', save=save_data)
# main(save=save_data)