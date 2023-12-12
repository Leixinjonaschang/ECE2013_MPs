# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def find_max(dict):
      max = 0
      maxkey = None
      for key in dict.keys():
                if dict[key] > max:
                        max = dict[key]
                        maxkey = key
      return maxkey

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    occurance = {}
    tags = {}
    for sentence in train:
        for pair in sentence:
                word = pair[0]
                tag = pair[1]
                if word in occurance.keys():
                        if tag in occurance[word].keys():
                                occurance[word][tag] +=1
                        else:
                                occurance[word][tag] = 1
                else:
                        occurance[word] = {}
                        occurance[word][tag] = 1
                
                if tag in tags.keys():
                        tags[tag] += 1
                else:
                        tags[tag] = 1
    
    tag_mode = find_max(tags)

    result = []
    for s_idx in range(len(test)):
        sentence = test[s_idx]
        result.append([])
        for w_idx in range(len(sentence)):
                word = sentence[w_idx]
                if word in occurance.keys():
                        tag = find_max(occurance[word])
                else:
                        tag = tag_mode
                result[s_idx].append((word,tag))
    print('baseline finished')
    return result