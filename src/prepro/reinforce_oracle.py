# Credits
# Written by Shashi Narayan to use original ROUGE
# Improved by Yang Liu to use a must faster ROUGE


import codecs
import itertools
import os
import re
import sys
import random
import math
from multiprocessing import Pool

from typing import List

import numpy

import prepro.fastRouge as rouge

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def cal_rouge(fullset, sentdata, golddata):
    """
    fullset: set of selected sentences for the summary
    sentdata: list of sentences of the document
    golddata: ground truth summary
    """
    fullset.sort()
    model_highlights = [sentdata[idx] for idx in range(len(sentdata)) if idx in fullset]
    rouge_1 = rouge.rouge_n(model_highlights, golddata, 1)['f']
    rouge_2 = rouge.rouge_n(model_highlights, golddata, 2)['f']
    # rouge_l = rouge.rouge_l_summary_level(model_highlights, golddata)['f']
    # rouge_score = (rouge_1 + rouge_2 + rouge_l) / 3.0
    rouge_score = (rouge_1 + rouge_2 ) / 3.0
    return rouge_score, fullset


def _multi_run_wrapper(args):
    return cal_rouge(*args)


def build_oracle_rewards(args, source, target, selected_sents):
    selected_sents = [i for i,label in enumerate(selected_sents) if label==1 ]

    pool = Pool(args.n_cpus)
    sent_limit = max(int (0.2 * len(source)) , len(selected_sents)) # tune
    p = min(int( 2 * sent_limit), len(source))
    print(' number of sentences::: %d , selection size: %d, p:%d' % (len(source), sent_limit, p))
    max_oracles = args.num_sample_rollout
    sentids_lst = [[sentid] for sentid in range(len(source))]
    arguments_list = []
    for sentids in sentids_lst:
        arguments_list.append((sentids, source, target))

    rougescore_sentwise = pool.map(_multi_run_wrapper, arguments_list)
    rougescore_sentwise.sort(reverse=True)

    top_rouge_sentences = numpy.array([item[1][0] for item in rougescore_sentwise[:p]] )# the id of the p best sentences with highest rouge score
    top_rouge_sentences.sort()
    rougescore_sentids = []
    rougescore_sentids += rougescore_sentwise[:p][:]
    arguments_list = []
    pool.close()
    pool.terminate()
    del pool
    pool = Pool(args.n_cpus)
    argument_set = set()
    combination_size = nCr(len(top_rouge_sentences), sent_limit)
    while len(argument_set) < min(max_oracles,combination_size) :
        subset_indices = sorted(random.sample(range(len(top_rouge_sentences)), min(sent_limit, len(top_rouge_sentences))))
        subset_indices_tuple =  tuple(subset_indices)
        if subset_indices_tuple not in argument_set:
            argument_set.add(subset_indices_tuple)
            arguments_list += [(top_rouge_sentences[subset_indices], source, target)]
    rougescore_sentids = pool.map(_multi_run_wrapper, arguments_list)
    rougescore_sentids.sort(reverse=True, key=lambda x: x[0])
    pool.terminate()
    pool.close()
    del pool
    return rougescore_sentids


