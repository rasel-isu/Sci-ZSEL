# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import re
import torch
import json
import sys
import io
import random
import time
import numpy as np
from blink.crossencoder.eval_utils import evaluate_cat_wise
from data_preparation.utils import read_jsonl
import wandb
from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# rasel
# from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_transformers.optimization import WarmupLinearSchedule
# from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import get_linear_schedule_with_warmup
# rasel

import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import EMA, CrossEncoderRanker, load_crossencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)


def evaluate(reranker, eval_dataloader, device, logger, context_length, zeshel=False, silent=True):
    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = 1 # len(WORLDS) (nhat: only 1 world)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    all_logits = []
    cnt = 0
    # nhat
    total_eval_loss = 0.0
    # nhat
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()
        # nhat
        total_eval_loss += eval_loss.item()  # nhat: accumulate eval loss
        # nhat
        tmp_eval_accuracy, eval_result = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        all_logits.extend(logits)

        nb_eval_examples += context_input.size(0)
        if zeshel:
            for i in range(context_input.size(0)):
                src_w = src[i].item()
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1


    # nhat
    avg_eval_loss = total_eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0.0
    logger.info(f"eval_loss : {avg_eval_loss}")  
    wandb.log({"eval_loss": avg_eval_loss})  # nhat: log to wandb
    # nhat

    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    if zeshel:
        macro = 0.0
        num = 0.0 
        for i in range(len(WORLDS)):
            if acc[i] > 0:
                acc[i] /= tot[i]
                macro += acc[i]
                num += 1
        if num > 0:
            logger.info("Macro accuracy: %.5f" % (macro / num))
            logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)
    else:
        if logger:
            logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)

    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    # nhat: log to wandb
    wandb.log({"test_accuracy": normalized_eval_accuracy})
    # nhat
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    # rasel
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
    )
    # rasel

    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler

def remove_special_chars(text):
    text = re.sub(r'[^A-Za-z0-9]', '', text)
    return text

def compare_with_dataset(train_data, candidate_input, label_input, tokenizer, missmatch_dict):
    train_kb_integer_ids = train_data["candidate_kb_integer_ids"]
    train_samples = utils.read_dataset('train', params["raw_data_path"])
    matched_all = True
    for o, l, kbid, ts in zip(candidate_input.tolist(), label_input.tolist(),train_kb_integer_ids.tolist(), train_samples):
        do = tokenizer.decode(o[l], skip_special_tokens=False)
        
        if ts['label_title']:
            label_title = ts['label_title'].lower()
            if label_title in missmatch_dict:
                return None
        oe_name_t = do.split('[unused2]')[0].replace('[CLS]', '').strip()
        label_def_t = remove_special_chars(do.split('[unused2]')[1].split('[SEP]')[0].strip())
        label_def = remove_special_chars(ts['label'].lower())
        
        # if ts['mention'] == 'hyperkeratosis':
        #     print(0)
        matched = False
        label_id_t = kbid[l]
        if ts['label_id'] == label_id_t:
            label_title_rem = remove_special_chars(label_title.strip())
            oe_name_t_rem = remove_special_chars(label_title.strip())

            if label_title_rem==oe_name_t_rem:
                # if label_def.strip()==label_def_t:
                matched = True
        if not matched:
            raise ValueError(f'data mismatched')

def main(params):
    wandb.init(dir="wandb_logs", project="eval_cross", 
            name=f'{params["output_path"]}-eval', resume="allow")

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])
    logger.info(json.dumps(params, indent=1))

    # Init model
    reranker = CrossEncoderRanker(params)
    device = reranker.device
    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    test_set_name = params["mode"]
    fname = os.path.join(params["data_path"], f"{test_set_name}.t7")
    # rasel
    if params['test_data_path'] != '':
        fname = os.path.join(params["test_data_path"], f"{test_set_name}.t7")
    #rasel
    valid_data = torch.load(fname)
    context_vecs = valid_data["context_vecs"]
    candidate_vecs = valid_data["candidate_vecs"]
    label_input = valid_data["labels"]
    candidate_kb_integer_ids = valid_data["candidate_kb_integer_ids"]
    sample_ids = valid_data["sample_ids"]
    if params["debug"]:
        max_n = 200
        context_vecs = context_vecs[:max_n]
        candidate_vecs = candidate_vecs[:max_n]
        label_input = label_input[:max_n]

    # compare_with_dataset(valid_data, candidate_input, label_input, reranker.tokenizer, {})

    context_input = modify(context_vecs, candidate_vecs, max_seq_length)
    if params["zeshel"]:
        src_input = valid_data["worlds"][:len(context_input)]
        valid_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        if params["save_trainable_data"]:
            valid_tensor_data = TensorDataset(context_input, label_input, 
            candidate_kb_integer_ids, sample_ids, 
            candidate_vecs, context_vecs)
        else:
            valid_tensor_data = TensorDataset(context_input, label_input, 
            candidate_kb_integer_ids, sample_ids)

    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )

    model = reranker.model

    ema = EMA(model)
    model = reranker.model
    epoch_output_folder_path= params["path_to_model"].replace('pytorch_model.bin', '')
    ema.apply(model) 
    reranker.model = model
    
    results = evaluate_cat_wise(
            params,
            test_set_name,
            epoch_output_folder_path,
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            zeshel=params["zeshel"],
            silent=params["silent"],
        )
    
if __name__ == "__main__":    

    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()
    args = parser.parse_args()
    # print(args)

    params = args.__dict__

    main(params)
