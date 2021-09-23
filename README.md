# LongScientificFormer

Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)


## Data Preparation 
### Step 1: download the processed data

[Pre-processed data](https://drive.google.com/file/d/1xYHXYoQBa7DJVrq0ePly58ioq2EmmVG8/view)

Put all files into `raw_data` directory

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-4.2.2` directory. 
#### step 3. extracting paragraphs from GROBID XML files

```
python preprocess.py -mode extract_pdf_paragraphs -log_file ../logs/extract_paragraphs.log
```

#### step 4. extracting text from TIKA XML files

```
python preprocess.py -mode get_text_clean_tika -log_file ../logs/extract_tika_text.log
```

#### step 5. Tokenize texts from papers and slides using stanfordCoreNLP

```
python preprocess.py -mode tokenize  -save_path ../temp -log_file ../logs/tokenize_by_corenlp.log
```


####  Step 6. Extract source, section, and target from tokenized files 

```
python preprocess.py -mode clean_paper_jsons -save_path ../media_json_data/  -n_cpus 30 -log_file ../logs/build_media_json.log
```


#### Step 7. Generate BERT `.pt` files from source, sections and targets

```
python preprocess.py -mode format_to_bert -raw_path ../media_json_data/ -save_path ../media_bert_data  -lower -n_cpus 40 -log_file ../logs/build_bert_files.log
```


## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Train

```
python train.py -ext_dropout 0.4 -lr 2e-1  -visible_gpus 3 -report_every 10 -save_checkpoint_steps 100 -batch_size 1 -train_steps 100000 -accum_count 10  -log_file ../logs/ext_bert -use_interval true -warmup_steps 100  
```
To continue training from a checkpoint
```
python train.py  -ext_dropout 0.1 -lr 2e-3  -train_from ../models/model_step_99000.pt -visible_gpus 1,2,3 -report_every 200 -save_checkpoint_steps 1000 -batch_size 1 -train_steps 100000 -accum_count 2  -log_file ../logs/ext_bert -use_interval true -warmup_steps 10000
```

To select the best model on validation set:
```
python train.py -mode validate -valid_batch_size 1 -log_file ../logs/ext_bert_valid -test_all True -model_path ../models -sep_optim true -use_interval true -visible_gpus 0 -alpha 0.95 -result_path ../results/ext ```
### Test

```
python train.py -mode test  -test_batch_size 1  -log_file ../logs/ext_bert_test -test_from ../models/model_step_99000.pt -model_path ../models -sep_optim true -use_interval true -visible_gpus 0 -alpha 0.95 -result_path ../results/ext 
```


