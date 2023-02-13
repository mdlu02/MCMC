# MCMC
Markov Chain Monte Carlo Substitution Cipher Decoder

This repository includes the code and text libraries to run a Markov Chain Monte Carlo (MCMC) algorithm for deciphering scrambled texts prepared using a substitution cipher. The algorithm is based on the Metropolis Hasings algorith, a variant of the MCMC class of algorithms. The algorithm approximates the substitution permutation used the encode the original text. 

Texts of type .txt or .pdf in \text_data are mined to build the transition and probability matricies. Text files in \encoded_texts are read and deciphered with the algorithm's outputs being saved in \decoded_texts.
