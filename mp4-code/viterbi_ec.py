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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import numpy as np
import math


def viterbi_ec(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    START_TAG = "START"
    END_TAG = "END"
    UNKNOWN = "UNKNOWN"

    # Part1: Count occurrences of tags, tag pairs, tag/word pairs

    # Number of sentences in training set
    tot_ini = 0

    alpha2 = 1e-5
    my_word_dict = {}
    hapax_count = 0
    hapax_words = []

    # Number of occur times of tags
    tagMat = {}

    for sentence in train:
        tot_ini += 1
        for i in range(len(sentence)):
            word = sentence[i][0]
            tag = sentence[i][1]
            if tag not in tagMat:
                tagMat[tag] = 1
            else:
                tagMat[tag] += 1

            if word not in my_word_dict:
                my_word_dict[word] = [1, tag]
            else:
                update_count = my_word_dict[word][0] + 1
                my_word_dict[word] = [update_count, tag]

    for my_word in my_word_dict:
        if my_word_dict[my_word][0] == 1:
            hapax_count += 1
            hapax_words.append(my_word)

    del tagMat[START_TAG]
    del tagMat[END_TAG]

    tagPair = {START_TAG: {}}
    twPair = {}
    for key in tagMat.keys():
        tagPair.setdefault(key, {})
        twPair.setdefault(key, {})

    wordList = [UNKNOWN]

    for sentence in train:
        for i in range(1, len(sentence) - 1):

            preTag = sentence[i - 1][1]
            word = sentence[i][0]
            tag = sentence[i][1]

            if tag not in tagPair[preTag]:
                tagPair[preTag][tag] = 1
            else:
                tagPair[preTag][tag] += 1

            if word not in twPair[tag]:
                twPair[tag][word] = 1
            else:
                twPair[tag][word] += 1

            if word not in wordList:
                wordList.append(word)

    num_hapax_tags = 0
    for w in wordList:
        for key in tagMat.keys():
            if w in hapax_words and (w in twPair[key]):
                num_hapax_tags += twPair[key][w]

    keyList = list(tagMat.keys())
    keyNum = len(keyList)

    # the probability that tag T occurs given that the word was hapax,P(T|hapax)
    P_tag_hapax = {}
    for my_key in tagMat.keys():
        curr_tag_count = 0
        for my_word2 in hapax_words:
            my_tag = my_word_dict[my_word2][1]
            if my_tag == my_key:
                curr_tag_count += 1
        P_tag_hapax[my_key] = (curr_tag_count + alpha2) / (num_hapax_tags + keyNum * alpha2)

    # Part2: Compute smoothed probabilities

    alpha = 1e-10  # smoothing factor

    initial = {}
    logIni = {}
    for key in tagMat.keys():
        if key in tagPair[START_TAG]:
            sub_ini = tagPair[START_TAG][key]
        else:
            sub_ini = 0

        # Probability = (N of tag pairs + alpha) / (total n of sentences + number of tags * alpha)
        p = (sub_ini + alpha) / (tot_ini + keyNum * alpha)
        initial.setdefault(key, p)
        logIni.setdefault(key, math.log(p))

    transition = {}
    logTrans = {}
    for key_1 in tagMat.keys():
        tot_trans = sum(tagPair[key_1].values())
        for key_2 in tagMat.keys():
            if key_2 in tagPair[key_1]:
                sub_trans = tagPair[key_1][key_2]
            else:
                sub_trans = 0
            pTrans = (sub_trans + alpha) / (tot_trans + keyNum * alpha)
            transition.setdefault(key_1, {}).update({key_2: pTrans})
            logTrans.setdefault(key_1, {}).update({key_2: math.log(pTrans)})

    # Emission probability = given a tag T P(W|T)
    emission = {}
    logEmiss = {}
    for w in wordList:
        if w == UNKNOWN:
            for key in tagMat.keys():
                pms = P_tag_hapax[key]
                emission.setdefault(key, {}).update({w: pms})
                logEmiss.setdefault(key, {}).update({w: math.log(pms)})
            continue
        else:
            tot_emiss = 0
            sub_emiss = [0 for i in range(keyNum)]
            num_emiss = -1
            for key in tagMat.keys():
                num_emiss += 1
                if w in twPair[key].keys():
                    sub_emiss[num_emiss] = twPair[key][w]
                    tot_emiss += twPair[key][w]
            num_emiss = -1
            for key in tagMat.keys():
                num_emiss += 1
                pEmiss = (sub_emiss[num_emiss] + alpha) / (tot_emiss + keyNum * alpha)
                emission.setdefault(key, {}).update({w: pEmiss})
                logEmiss.setdefault(key, {}).update({w: math.log(pEmiss)})

    # # Part3: Take the log of each probability

    logIni = {}
    for key in initial.keys():
        logIni[key] = np.log(initial[key])

    logTrans = {}
    for key_1 in transition.keys():
        # for key_2 in transition.keys():
        for key_2 in transition[key_1].keys():
            # logTrans[key_1][key_2] = np.log(transition[key_1][key_2])
            num1 = math.log(transition[key_1][key_2])
            logTrans.setdefault(key_1, {}).update({key_2: num1})

    logEmiss = {}
    for key in emission.keys():
        for w in emission[key].keys():
            # logEmiss[key][w] = np.log(emission[key][w])
            num2 = math.log(emission[key][w])
            logEmiss.setdefault(key, {}).update({w: num2})
    # Part4: Construct the trellis. Notice that for each tag/time pair,
    # you must store not only the probability of the best path but also
    # a pointer to the previous tag/time pair in that path.

    retMat = []
    for item in test:
        N = keyNum
        T = len(item) - 2
        Viterbi = np.zeros((N, T))
        Backpointer = np.zeros((N, T))

        for n in range(N):
            key = keyList[n]
            word = item[1]
            if word not in wordList:
                word = UNKNOWN
            Viterbi[n][0] = logIni[key] + logEmiss[key][word]
            Backpointer[n][0] = 0

        for t in range(1, T):
            for n in range(N):
                key = keyList[n]
                word = item[t + 1]
                if word not in wordList:
                    word = UNKNOWN
                viterbi = -math.inf
                backpointer = 0
                for nPast in range(N):
                    keyPast = keyList[nPast]
                    ProSum = Viterbi[nPast][t - 1] + logTrans[keyPast][key] + logEmiss[key][word]
                    if ProSum > viterbi:
                        viterbi = ProSum
                        backpointer = nPast
                Viterbi[n][t] = viterbi
                Backpointer[n][t] = backpointer

        # BestPro = [np.max(Viterbi[:,-1])]
        TagIndex = np.argmax(Viterbi[:, -1])
        retSent = [(item[-2], keyList[TagIndex])]
        Pointer = Backpointer[TagIndex, -1]
        for t in range(T - 2, -1, -1):
            TagIndex = int(Pointer)
            outTuple = (item[t + 1], keyList[TagIndex])
            retSent.append(outTuple)
            Pointer = Backpointer[TagIndex, t]
        retSent.append(('START', 'START'))
        retSent.reverse()
        retSent.append(('END', 'END'))
        retMat.append(retSent)

    # Part5: Return the best path through the trellis

    return retMat
