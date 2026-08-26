# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from copy import deepcopy
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np
from utils import compare_with_multiple_gt, compare_with_multiple_gt_with_altid
import wandb

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import get_linear_schedule_with_warmup

import blink.candidate_retrieval.utils
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
from performence import evaluate_biencoder_and_crossencoder


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

def get_gt_rank(kb, sorted_cand_score, gt_list):
    c = 1
    for cand_score in sorted_cand_score:
        cand_id = cand_score['id']
        is_matched, matched_gt = compare_with_multiple_gt(cand_score, gt_list)
        if is_matched:
            return {**matched_gt, **{'score':cand_score['score'], 'rank':c, 'candt':cand_id}}
        if not is_matched:
            is_matched, matched_gt, altid = compare_with_multiple_gt_with_altid(kb, cand_score, gt_list)
            if is_matched:
                return {**matched_gt, **{'score':cand_score['score'], 'rank':c, 'candt':cand_id,'altdiseaseid':altid}}
                
        c+=1

def get_gt_rank_train_set(sorted_cand_score, gt):
    c = 1
    for cand_score in sorted_cand_score:
        cand_id = cand_score['int_id']
        is_matched = False
        if cand_id == gt:
            is_matched = True
        matched_gt = {}
        # is_matched, matched_gt = compare_with_multiple_gt(cand_score, gt_list)
        if is_matched:
            return {**matched_gt, **{'score':cand_score['score'], 'rank':c, 'candt':cand_id}}
     
        c+=1

def get_cand_score_for_train_set(map_dict, kb, kb_int_id, logits, label_ids, sample_ids):
    all_cand_score = []
    all_linked = []
    for each_id, each_score, each_label_indx, each_smpl_id in zip(kb_int_id, logits, label_ids, sample_ids):
        cand_score = []
        for id, score in zip(each_id, each_score):
            cand_score.append({'int_id':int(id), 'id':map_dict[str(id)], 'score':score})
        sorted_cand_score = sorted(cand_score, key=lambda x: x['score'], reverse=True)
        actual_label = int(each_id[each_label_indx])
        gt_rank = get_gt_rank_train_set(sorted_cand_score, actual_label)
        if gt_rank['rank']==1:
            all_linked.append(True) 
        else:
            all_linked.append(False) 
        all_cand_score.append({'reranked_for':each_smpl_id,'reranker_gt_rank':gt_rank, 'sorted_cand_score':sorted_cand_score})

    return all_cand_score, all_linked

def get_cand_score_for_test_set(grag_data, map_dict, kb, kb_int_id, logits, label_ids, sample_ids):
    all_cand_score = []
    all_linked = []
    for each_id, each_score, each_label_indx, each_smpl_id in zip(kb_int_id, logits, label_ids, sample_ids):
        cand_score = []
        for id, score in zip(each_id, each_score):
            # if str(id) == '-1':
            #     cand_score.append({'int_id':int(id), 'id':str(id), 'score':score})
            # else:
            cand_score.append({'int_id':int(id), 'id':map_dict[str(id)], 'score':score})
        sorted_cand_score = sorted(cand_score, key=lambda x: x['score'], reverse=True)


        actual_label = grag_data[each_smpl_id]['ground_truth']
        gt_rank = get_gt_rank(kb, sorted_cand_score, actual_label)
        
        # if gt_rank:
        if gt_rank['rank']==1:
            all_linked.append(True) 
        else:
            all_linked.append(False) 
        # else:
        #     all_linked.append(False) 


        all_cand_score.append({'reranked_for':each_smpl_id,'reranker_gt_rank':gt_rank, 'sorted_cand_score':sorted_cand_score})

    return all_cand_score, all_linked


    
def get_candidate_vecs_as_crosencoder_predicts():
    pass

def evaluate_cat_wise(params, test_set_name, output_path, 
                      reranker, 
    eval_dataloader, device, logger, context_length, zeshel=False, silent=True):
    
    # params = deepcopy(params)
    # params["path_to_model"] = output_path + '/pytorch_model.bin'
    # # params["dropout_rate"] = 0.0
    # reranker = CrossEncoderRanker(params)

    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    # rasel
    samples_map = {}
    with open(f'{params["kb_file_path"]}') as f:
        exact_kb = json.load(f)
        if params["onto"] == 'ncbi':
            kb = {}
            for e in exact_kb:
                ent = exact_kb[e]
                kb[e] = ent
                for i in ent['altdiseaseid']:
                    kb[i] = ent
        elif params["onto"] in ['bc5cdr', 'cmo', 'vt', 'lpt', 'cometa']:
            kb = deepcopy(exact_kb)

    with open(f'{params["raw_data_path"]}/id_map.json') as f:
        map_dict = json.load(f)
        swapped_map_dict = {v: k for k, v in map_dict.items()}

    grag_filename = f'{params["mode"]}_grag.json'
    if params["mode"] == 'valid':
        grag_filename = f'test_grag.json'

    with open(f'{params["grag_data_path"]}/{grag_filename}') as f:
        raw_grag_data = json.load(f)
        grag_data={}
        for i in raw_grag_data:
            gt_list = i['ground_truth']
            gt_with_mapped_id = []
            for gt in gt_list:
                if gt['id'] in swapped_map_dict:
                    gt['map_id'] = int(swapped_map_dict[gt['id']])
                else:
                    gt['map_id'] = None

                gt_with_mapped_id.append(gt)
            i['ground_truth'] = gt_with_mapped_id
            grag_data[i['sample_id']] = i
    # rasel
    candidate_pool = torch.load(params["cand_pool_path"])

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0
    all_cand_score = []
    count_linked = 0
    all_gt_rank = []
    candidate_kb_integer_ids_crossenc = []
    all_logits = []

    # to store 
    nn_context = []
    nn_candidates = []
    nn_labels = []
    cands = []
    prime_candidate_ids = []
    sample_ids_list = []
    # to store 

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
        kb_int_id = batch[2]
        sample_ids = batch[3]

        

        # # rasel : Check data
        # decoded_text_context_input = ''
        # tokenizer = reranker.tokenizer
        # for contx in context_input:
        #     for can in contx:
        #         decoded_text = tokenizer.decode(can, skip_special_tokens=False)
        #         decoded_text_context_input+= f"{decoded_text}\n\n"
        # with open('decoded_text_context_input.txt', 'w') as f:
        #     f.write(decoded_text_context_input + f'\n\n{kb_int_id}')
        # #rasel


        with torch.no_grad(): 
            reranker_out  = reranker(context_input, label_input, context_length)
            eval_loss, logits = reranker_out[0], reranker_out[1]

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()
        

        # nhat
        total_eval_loss += eval_loss.item()  # nhat: accumulate eval loss
        # nhat

        # rasel
        kb_int_id = kb_int_id.cpu().numpy()
        sample_ids = sample_ids.cpu().numpy()
        if params["save_trainable_data"]:
            cand_score, linked = get_cand_score_for_train_set(map_dict, kb, kb_int_id, logits, 
            label_ids, sample_ids)
            candidate_vecs = batch[4]
            context_vecs = batch[5]
            for i, pred_cand in enumerate(cand_score):
                cand_kb_int_id_crossenc = [pred_int_id['int_id'] for pred_int_id in pred_cand['sorted_cand_score']]
                nn_context.append(context_vecs[i].cpu().tolist())
                cur_candidates = candidate_pool[cand_kb_int_id_crossenc]
                nn_candidates.append(cur_candidates.cpu().tolist())
                prime_candidate_ids.append(cur_candidates.cpu().tolist())
                crossenc_label_pointer = pred_cand['reranker_gt_rank']['rank']-1
                nn_labels.append(crossenc_label_pointer)
                cands.append(cand_kb_int_id_crossenc)
                sample_ids_list.append(sample_ids[i])
        else:
            cand_score, linked = get_cand_score_for_test_set(grag_data, map_dict, kb, 
                                                         kb_int_id, logits, label_ids, sample_ids)
        for l in linked:
            if l:
                count_linked+=1
        all_cand_score.extend(cand_score)
        # rasel

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
            logger.info(f"Linked accuracy: {count_linked / nb_eval_examples}")


    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    results["cross_prediction"] = all_cand_score

    # rasel
    os.makedirs(output_path, exist_ok=True)
    cross_pred_file = output_path+'/crossencoder_predictions.json'
    with open(cross_pred_file, 'w') as f:
        json.dump(results['cross_prediction'], f, default=str, indent=1)

    fname = os.path.join(params["data_path"], f'{test_set_name}.t7')
    if params['test_data_path'] != '':
        fname = os.path.join(params["test_data_path"], f"{test_set_name}.t7")

    bi_pred_file = fname.replace('.t7', '.json')
    eval_result = evaluate_biencoder_and_crossencoder(params, 
                                        bi_pred_file, 
                                        cross_pred_file, split_name=test_set_name)
    
    results["recall_at_1"] = eval_result['eval_cross']['recall_at_1']

    if params["save_trainable_data"]:
        nn_context = torch.LongTensor(nn_context)
        nn_candidates = torch.LongTensor(nn_candidates)
        nn_labels = torch.LongTensor(nn_labels)
        nn_cands_id = torch.LongTensor(cands)
        prime_candidate_ids_tensor = torch.LongTensor(prime_candidate_ids)
        sample_ids_tensor = torch.LongTensor(sample_ids_list)
        nn_data = {
            'context_vecs': nn_context,
            'candidate_vecs': nn_candidates,
            'labels': nn_labels,
            'candidate_kb_integer_ids': nn_cands_id,
            'prime_candidates': prime_candidate_ids_tensor,
            'sample_ids': sample_ids_tensor
        }
        save_data_dir = os.path.join(
            params['output_path'],
            "top%d_candidates" % params['top_k'],
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % params['mode'])
        torch.save(nn_data, save_data_path)
    # rasel

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

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler
