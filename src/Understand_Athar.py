import json

import numpy
import numpy as np
from models.data_loader import *
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_data_statistics():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bert_data_path", default='../bert_data_oracle_rewards/')
    args = parser.parse_args()

    article_tokens = []
    article_sentences = []
    slide_tokens = []
    morethan5000 = 0
    sent_sizes = []
    maxlen = -1
    for pt_file in load_dataset(args, 'train', shuffle=True):
        # print('pt_file size:', len(pt_file))
        for ex in pt_file:
            src = ex['src']
            clss = numpy.array(ex['clss'])
            labels = ex['src_sent_labels']
            reward_oracle = ex['reward_oracle']
            print('-------------------------------------------------------')
            for reward, oracle in reward_oracle:
                print(reward, len(oracle), oracle)
            continue
            # print(sum(labels),int(0.2* len(clss)), len(clss))
            # generates the sentence lengths
            lengths = numpy.append(clss[1:] - clss[:-1] , len(src)- clss[-1])
            maxlen = max(maxlen, max(lengths))

            np_src = np.array(src)
            cls_indices = np.argwhere(np_src==101)
            prev_cls = cls_indices[0][0]
            for index in cls_indices[1:]:
                sent_sizes.append(index[0]-prev_cls)
                prev_cls = index[0]
            tgt = ex['tgt']
            src_sent_labels = ex['src_sent_labels']
            if len(tgt) > 5000:
                morethan5000 += 1
            # print('NUMBER OF POS labels:::::', len(src), len(src_sent_labels), sum(src_sent_labels))
            slide_tokens.append(len(tgt))
            article_tokens.append(len(src))
            article_sentences.append(len(src_sent_labels))
    print(maxlen)
    sent_size_np = np.array(sent_sizes)
    big_sents = np.argwhere(sent_size_np > 200).size
    print('---- number of sents greater than 100: {}, total sentences: {}, ration:{}'.format(big_sents, sent_size_np.size,
                                                                   big_sents / sent_size_np.size))
    print('--- number of slides with more than 5000 tokens:', morethan5000)
    print('---- max tokens counts :::', max(article_tokens))
    print('---- max sentence counts :::', max(article_sentences))

    def plot_token_counts(article_tokens):
        n, bins, patches = plt.hist(x=article_tokens, bins='auto', color='purple',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('token counts')
        plt.ylabel('Frequency')
        plt.title('Token Count Per Article')
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()

    def plot_sent_sizes(sent_sizes):
        n, bins, patches = plt.hist(x=sent_sizes, bins='auto', color='purple',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('sentence token count')
        plt.ylabel('Frequency')
        plt.title('Token Count Per Sentence')
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.xlim(xmax=200)
        plt.show()

    #plot_sent_sizes(sent_sizes)
    plot_token_counts(slide_tokens)


def plot_xent_results():
    xents = [float(line.split()[-7][:-1]) for line in open('../xents.txt', 'r').readlines()]
    import matplotlib.pyplot as plt
    import numpy as np
    print(xents)
    plt.plot(xents)
    plt.xlabel('step')
    plt.ylabel('Xent Loss')
    plt.title('Loss')
    #plt.text(23, 45, r'$\mu=15, b=3$')
    #maxfreq = n.max()
    # Set a clean upper y-axis limit.
    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


# plot_xent_results()
plot_data_statistics()
