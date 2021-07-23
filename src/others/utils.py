import os
import re
import shutil
import time
from datasets import load_metric


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def test_rouge(cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print('*' * 50)
    print('There are %d reference summaries and %d candidate summaries that do not match'%(len(references), len(candidates)))
    assert (len(candidates) == len(references)), f" There are {len(references)} reference summaries and {len(references)} candidate summaries that do not match"
    metric = load_metric('rouge')
    metric.add_batch(predictions=candidates, references=references)
    rouge_scores = metric.compute()
    return rouge_scores


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict['rouge1'].high.fmeasure * 100,
        results_dict['rouge2'].high.fmeasure * 100,
        results_dict['rougeL'].high.fmeasure * 100,

        results_dict['rouge1'].high.recall * 100,
        results_dict['rouge2'].high.recall * 100,
        results_dict['rougeL'].high.recall * 100

    )