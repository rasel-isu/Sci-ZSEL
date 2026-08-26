import argparse
import os

from tqdm import tqdm
from eval import cat_eval

def rewirite_report(corpus,raw_data_path, pred_file, test_data_filepath, eval_before_fine_tune):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default=f"{raw_data_path}train.json")
    parser.add_argument("--dev_kb", default=f"kb/{corpus}_kb.json")
    parser.add_argument("--lego_test_data", default=f"{test_data_filepath}")
    parser.add_argument("--eval_before_fine_tune", default=eval_before_fine_tune)
    args = parser.parse_args()
    cat_eval(args, None, pred_file, kbpath=args.dev_kb)

def make_report(corpus, sptitname):

    # raw_data_path = f'data/blink/zshel_64_context_length/{corpus}/{sptitname}/prime/'
    # output_path = f'model_disambiguation/{corpus}/{sptitname}/prime/before_fine_tune/ncbi_pred_prime_train_with_test_title.json'
    # test_data_filepath = f'data/blink/zshel_64_context_length/{corpus}/{sptitname}/test.json'
    # eval_before_fine_tune = True
    # rewirite_report(corpus,raw_data_path, output_path, test_data_filepath, eval_before_fine_tune)

    # exps = os.listdir(f'model_disambiguation/{corpus}/{sptitname}/')
    exps = ['original_def',
       'prime','original_def_small_set', 'ho_prime_others_not'
    ]

    for exp in tqdm(exps):
      if '.json' in exp or '.pptx' in exp:
         continue
      epochs = [i for i in range(1, 11)]
      for epoch in epochs:
        raw_data_path = f'data/blink/zshel_64_context_length/{corpus}/{sptitname}/{exp}/'
        output_path = f'model_disambiguation/{corpus}/{sptitname}/{exp}/zeshel_disambiguation_attention_{exp}_{corpus}_{epoch}pred/.json'
        test_data_filepath = f'data/blink/zshel_64_context_length/{corpus}/{sptitname}/test.json'
        eval_before_fine_tune = False
        rewirite_report(corpus,raw_data_path, output_path, test_data_filepath, eval_before_fine_tune)

make_report('ncbi', 'train')
# make_report('bc5cdr', 'train')

'model_disambiguation/ncbi/train/original_def/zeshel_disambiguation_attention_original_def_ncbi_1pred/.json'
'model_disambiguation/ncbi/train/original_def/zeshel_disambiguation_attention_original_def_ncbi_epoch_1pred/.json'