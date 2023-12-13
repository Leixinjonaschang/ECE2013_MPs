import math
import numpy as np


def viterbi_ec(train, test):
    START_TAG = "START"
    END_TAG = "END"
    UNKNOWN = "UNKNOWN"

    # Part1: Count occurrences of tags, tag pairs, and tag/triple pairs
    tot_ini = 0
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

    del tagMat[START_TAG]
    del tagMat[END_TAG]

    tagTriple = {}  # New dictionary to store triples
    twPair = {}
    for key in tagMat.keys():
        twPair.setdefault(key, {})

    wordList = [UNKNOWN]

    for sentence in train:
        for i in range(2, len(sentence) - 1):
            prevTag = sentence[i - 2][1]
            curTag = sentence[i - 1][1]
            nextTag = sentence[i][1]

            if (prevTag, curTag) not in tagTriple:
                tagTriple[(prevTag, curTag)] = {nextTag: 1}
            else:
                tagTriple[(prevTag, curTag)][nextTag] = tagTriple[(prevTag, curTag)].get(nextTag, 0) + 1

            word = sentence[i][0]
            tag = sentence[i][1]

            if word not in twPair[tag]:
                twPair[tag][word] = 1
            else:
                twPair[tag][word] += 1

            if word not in wordList:
                wordList.append(word)

    # Part2: Compute smoothed probabilities

    keyList = list(tagMat.keys())
    keyNum = len(keyList)
    alpha = 1e-10  # smoothing factor

    # Transition probabilities
    transition = {}
    logTrans = {}
    # 为每个可能的第一个和第二个标签添加与 START_TAG 相关的转移概率
    for key_1 in tagMat.keys():
        tot_trans = sum([tagTriple[(START_TAG, k1)].get(k2, 0) for k1, k2 in tagTriple.keys() if k1 == key_1])
        for key_2 in tagMat.keys():
            sub_trans = sum(
                [tagTriple[(START_TAG, k1)].get(k2, 0) for k1, k2 in tagTriple.keys() if k1 == key_1 and k2 == key_2])
            pTrans = (sub_trans + alpha) / (tot_trans + keyNum * alpha)
            transition.setdefault((START_TAG, key_1), {}).update({key_2: pTrans})
            logTrans.setdefault((START_TAG, key_1), {}).update({key_2: math.log(pTrans)})

    for key_1 in tagMat.keys():
        for key_2 in tagMat.keys():
            tot_trans = sum([tagTriple[(k1, k2)].get(key_2, 0) for k1, k2 in tagTriple.keys() if k1 == key_1])
            for key_3 in tagMat.keys():
                sub_trans = sum(
                    [tagTriple[(k1, k2)].get(key_3, 0) for k1, k2 in tagTriple.keys() if k1 == key_1 and k2 == key_2])
                pTrans = (sub_trans + alpha) / (tot_trans + keyNum * alpha)
                transition.setdefault((key_1, key_2), {}).update({key_3: pTrans})
                logTrans.setdefault((key_1, key_2), {}).update({key_3: math.log(pTrans)})

    # Emission probability = given a tag T P(W|T)
    emission = {}
    logEmiss = {}
    for w in wordList:
        for key in tagMat.keys():
            tot_emiss = sum(twPair[key].values())
            sub_emiss = twPair[key].get(w, 0)
            pEmiss = (sub_emiss + alpha) / (tot_emiss + keyNum * alpha)
            emission.setdefault(key, {}).update({w: pEmiss})
            logEmiss.setdefault(key, {}).update({w: math.log(pEmiss)})

    # Part3: Construct the trellis
    retMat = []
    for item in test:
        N = keyNum
        T = len(item) - 2
        Viterbi = np.zeros((N, N, T))
        Backpointer1 = np.zeros((N, N, T), dtype=int)
        Backpointer2 = np.zeros((N, N, T), dtype=int)

        # Initialize the Viterbi matrix
        for n1 in range(N):
            for n2 in range(N):
                firstTag = keyList[n1]
                secondTag = keyList[n2]
                word = item[2]
                if word not in wordList:
                    word = UNKNOWN
                Viterbi[n1][n2][0] = logTrans[(START_TAG, firstTag)][secondTag] + logEmiss[secondTag][word]
                Backpointer1[n1][n2][0] = -1
                Backpointer2[n1][n2][0] = -1

        # Dynamic programming part
        for t in range(1, T):
            for n1 in range(N):
                for n2 in range(N):
                    key = keyList[n2]
                    word = item[t + 2]
                    if word not in wordList:
                        word = UNKNOWN
                    viterbi = -math.inf
                    backpointer = (0, 0)
                    for nPast1 in range(N):
                        for nPast2 in range(N):
                            keyPast1 = keyList[nPast1]
                            keyPast2 = keyList[nPast2]
                            ProSum = Viterbi[nPast1][nPast2][t - 1] + logTrans[(keyPast1, keyPast2)][key] + \
                                     logEmiss[key][word]
                            if ProSum > viterbi:
                                viterbi = ProSum
                                backpointer1, backpointer2 = nPast1, nPast2
                    Viterbi[n1][n2][t] = viterbi
                    Backpointer1[n1][n2][t] = backpointer1
                    Backpointer2[n1][n2][t] = backpointer2

        # Backtracking for the best path
        TagIndex1, TagIndex2 = np.unravel_index(np.argmax(Viterbi[:, :, -1]), (N, N))
        retSent = []
        Pointer1 = Backpointer1[TagIndex1, TagIndex2, -1]
        Pointer2 = Backpointer2[TagIndex1, TagIndex2, -1]
        for t in range(T - 2, -1, -1):
            TagIndex1, TagIndex2 = int(Pointer1), int(Pointer2)
            outTuple = (item[t + 2], keyList[TagIndex2])
            retSent.append(outTuple)
            Pointer1 = Backpointer1[TagIndex1, TagIndex2, t]
            Pointer2 = Backpointer2[TagIndex1, TagIndex2, t]
        retSent.reverse()

        retMat.append(retSent)

    return retMat
