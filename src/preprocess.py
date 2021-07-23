import argparse
import time

from others.log import init_logger
from prepro import data_builder


def do_format_to_lines(args):
    print(time.clock())
    data_builder.format_to_lines(args)
    print(time.clock())


def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())


def do_tokenize(args):
    print(time.clock())
    data_builder.tokenize(args)
    print(time.clock())


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-select_mode", default='greedy', type=str)
    parser.add_argument("-map_path", default='../data/')
    parser.add_argument("-raw_path", default='../line_data')
    parser.add_argument("-save_path", default='../data/')

    parser.add_argument("-shard_size", default=50, type=int)
    parser.add_argument('-min_src_nsents', default=20, type=int)
    parser.add_argument('-max_src_nsents', default=500, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=50, type=int)  # change to 50
    parser.add_argument('-min_tgt_ntokens', default=50, type=int)
    parser.add_argument('-max_tgt_ntokens', default=15000, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-log_file', default='../logs/slide_gen.log')

    parser.add_argument('-dataset', default='')

    parser.add_argument('-n_cpus', default=2, type=int)

    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder.' + args.mode + '(args)')

# pre-process
# step 1 command (extract sections from GROBID output): python preprocess.py -mode extract_pdf_sections -log_file ../logs/extract_section.log
# step 2 command (extract text from TIKA xml files): python preprocess.py -mode get_text_clean_tika -log_file ../logs/extract_tika_text.log
# step 3 command (sentence tokenize with stanfordCoreNLP): python preprocess.py -mode tokenize  -save_path ../temp -log_file ../logs/tokenize_by_corenlp.log
# step 4 command: python preprocess.py -mode clean_paper_jsons -save_path ../json_data/  -n_cpus 10 -log_file ../logs/build_json.log
# step 5 command: python preprocess.py -mode format_to_bert -raw_path ../json_data/ -save_path ../bert_data  -lower -n_cpus 40 -log_file ../logs/build_bert_files.log


# train
# python3 train.py  -ext_dropout 0.1 -lr 2e-3  -visible_gpus -1 -report_every 200 -save_checkpoint_steps 1000 -batch_size 1 -train_steps 100000 -accum_count 2  -log_file ../logs/ext_bert -use_interval true -warmup_steps 10000
# test
# python3 train.py  -mode test -batch_size 1 -test_batch_size 1 -bert_data_path ../bert_data -log_file  ../logs/ext_bert_test  -test_from ../models/model_step_99000.pt  -model_path ../models -sep_optim true -use_interval true -visible_gpus 0  -alpha 0.95 -result_path ../results/ext
