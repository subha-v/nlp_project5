from unicodedata import normalize
from utils import *
import random
import torch
SOS_TOKEN = 0
EOS_TOKEN = 1

# we're taking the part of the dataset thats 10 or some word s less and starts with english_prefixes


english_prefixes = ("i am", "he is", "i m ", "he is", "he s " "she is", "she s", 
"you are", "you re", "we are", "we re" "they are", "they re")

class Vocabulary:
    def __init__(self, language_name):
        self.language_name = language_name
        self.word_to_index = {"EOS": EOS_TOKEN, "SOS": SOS_TOKEN}
        self.word_to_count = {"EOS": 1, "SOS":1}
        self.index_to_word = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.num_words = 2
        

    def add_word(self, word):
        #only add a word to the dictionary if its not in it
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.word_to_count[word] = 1
            self.index_to_word[self.num_words] = word # inverse mapping
            self.num_words += 1
    def add_phrase(self, phrase):
        words = phrase.split(" ")
        for word in words:
            self.add_word(word)

class TranslationDataset:
    def __init__(self, dataset_path, language_name1, language_name2):
        self.dataset_path = dataset_path
        self.max_length = 10 #max length of a dataset since our thing is large
        lines = open(self.dataset_path, encoding ='utf-8').read().strip().split("\n")
        # this creates a list of all of the words
        # the split function creates a list of all the words
        # the delimiter between the english and french is the tab
        # we want to split on the tab therefore
        # we want to create english french pairs
        pairs = [[normalize_word(word) for word in line.split('\t')] for line in lines]
        #this normalizes all the words in the english and french pairs
        # so for example ['go', 'va !']
        # the outer list comprehension does this for all the pairs in the dataset
        self.input_vocab = Vocabulary(language_name1)
        self.target_vocab = Vocabulary(language_name2)
        #elem for elem in lst if elem %2 == 0
        self.pairs = [pair for pair in pairs if self.is_simple_sentence(pair)]
        # this makes the pairs only the one with simple sentences
        for input_pair, target_pair in pairs:
            self.input_vocab.add_phrase(input_pair)
            self.target_vocab.add_phrase(target_pair)
            #this adds english and french words tot he inuput vocab and target vocab classes




    def is_simple_sentence(self, pair):
        cond1 = len(pair[0].split(' '))< self.max_length #this checks if there are more than 10 words
        cond2 = len(pair[1].split(' ')) < self.max_length # this checks if there are more than 10 words int he french sentence
        cond3 = pair[0].startswith(english_prefixes) # using starts with to show if it starts with an english prefixes

        return(cond1 and cond2 and cond3) #returns true or false based on this 

    def sentenceToIndicies(self, sentence, vocab):
        indicies = []


        words = sentence.split(" ")
        for word in words:
            idx = vocab.word_to_index[word]
            indicies.append(idx)

        indicies.append(EOS_TOKEN)
        return indicies



    def get_random_sample(self):
        rand_idx = random.randint(0,len(self.pairs)-1) # this gets a random pair
        pair = self.pairs[rand_idx]
        lang1_indicies = self.sentenceToIndicies(pair[0], self.input_vocab)
        lang2_indicies = self.sentenceToIndicies(pair[1], self.target_vocab)

        lang1_idx_tensor = torch.tensor(lang1_indicies,dtype = torch.long)
        lang2_idx_tensor = torch.tensor(lang2_indicies, dtype = torch.long)

        return lang1_idx_tensor, lang2_idx_tensor

        # making them into tensors


        


        












