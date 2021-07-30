import bisect
import gc
import glob
import random

import torch

from others.log import logger


class Batch(object):

    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_sections = [x[1] for x in data]
            pre_token_sections = [x[2] for x in data]
            pre_tgt = [x[3] for x in data]
            pre_segs = [x[4] for x in data]
            pre_clss = [x[5] for x in data]
            pre_src_sent_labels = [x[6] for x in data]
            pre_section_ids = [x[7] for x in data]
            # todo

            src = torch.tensor(self._pad(pre_src, 0)).to(int)
            tgt = torch.tensor(self._pad(pre_tgt, 0)).to(int)
            segs = torch.tensor(self._pad(pre_segs, 0)).to(int)
            token_sections = torch.tensor(self._pad(pre_token_sections, 0)).to(int)
            mask_src = ~ (src == 0).to(int)
            mask_tgt = ~ (tgt == 0)

            clss = torch.tensor(self._pad(pre_clss, -1)).to(int)
            section_ids = torch.tensor(self._pad(pre_section_ids, -1)).to(int)
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0)).to(int)
            sections = torch.tensor(self._pad(pre_sections, 0)).to(int)
            mask_cls = ~ (clss == -1)
            clss[clss == -1] = 0
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))
            setattr(self, 'sections', sections.to(device))
            setattr(self, 'section_ids', section_ids.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'token_sections', token_sections.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))

            if is_test:
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        # logger.info('Loading %s dataset from %s, number of examples: %d' %
        #             (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '/' + corpus_type + '.[0-9]*.bert.pt'))
    if pts:
        if shuffle:
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '/' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def ext_batch_size_fn(new, count):
    if len(new) == 4:
        pass
    src, labels = new[0], new[4]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        # I think datasets contains train.0 train.1 .... and dataset_iter iterates over the indices of the train
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args,
                            dataset=self.cur_dataset, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']

        segs = ex['segs']
        if not self.args.use_interval:
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        sections = ex['sections']
        token_sections = ex['token_sections']
        section_ids = ex['section_ids']
        # todo check if section is working properly
        max_pos = self.args.max_pos
        max_sent_id = bisect.bisect_left(clss, max_pos)
        max_sec_id = bisect.bisect_left(section_ids, max_pos)
        # 30523 is the id of the [SEC]
        if len(src) > self.args.max_pos-1 and src[self.args.max_pos-1] == 101: # ends with [CLS]
            max_pos -= 1
            max_sent_id -=1
            if src[max_pos-1] == 30523: # ends with [SEC][CLS]
                max_pos -=1
                max_sec_id -=1
        if len(src) > self.args.max_pos-1 and src[self.args.max_pos-1] == 30523: # [SEC]
            max_pos -= 1
            max_sec_id -=1
        src = src[:max_pos]
        src[-1] = 102
        segs = segs[:max_pos]
        token_sections = token_sections[:max_pos]


        section_ids = section_ids[:max_sec_id]
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        sections = sections[: max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if is_test:
            return src, sections, token_sections, tgt, segs, clss, src_sent_labels, section_ids, src_txt, tgt_txt,
        else:
            return src, sections, token_sections, tgt, segs, clss, src_sent_labels, section_ids

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if len(ex['src']) == 0:
                continue
            ex = self.preprocess(ex, self.is_test)
            if ex is None:
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()  # just shuffles the dataset which is for example train.0
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                if len(b) == 0:
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)
                yield batch
            return
