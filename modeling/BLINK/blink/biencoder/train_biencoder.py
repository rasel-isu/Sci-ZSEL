# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from copy import deepcopy
import math
import os
import argparse
import pickle
from category_eval import MultiGTEvaluation
import torch
import json
import sys
import io
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import get_linear_schedule_with_warmup

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery

logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        try:
            context_input, candidate_input, _, actual_label = batch
        except ValueError as e:
            print(f'No "world" : {e}\n so ignored one _')
            if len(batch) == 3:
                context_input, candidate_input, actual_label = batch

        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()

        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(
                torch.arange(params["eval_batch_size"])
        ).numpy()

        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results

def evaluate_with_all_candidate(
    reranker, eval_dataloader, all_candidate_encoding, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        try:
            context_input, candidate_input, _, actual_label, mention_id = batch
        except ValueError as e:
            # print(f'No "world" : {e}\n so ignored one _')
            if len(batch) == 4:
                context_input, candidate_input, actual_label, mention_id = batch



        with torch.no_grad():
            # eval_loss, logits = reranker(context_input, candidate_input)
            scores = reranker.score_candidate(
                    context_input, None, cand_encs=all_candidate_encoding.to(device)
                )
            scores, indicies = scores.topk(64)
            scores = scores.data.cpu()
            indicies = indicies.data.cpu().tolist()

        actual_label = actual_label.data.cpu().tolist()

        tmp_eval_accuracy = 0
        for al, p_ind in zip(actual_label, indicies):
            if al[0] in p_ind:
                tmp_eval_accuracy+=1

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
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

    # num_train_steps = int(len_train_data / batch_size / grad_acc) * 5
    # num_train_steps = len_train_data // 400

    # num_train_steps = max(2, math.ceil((len_train_data - 100) / 500))
    # num_train_steps  = max(2, math.ceil((len_train_data - 100) / 500)) * epochs
    # num_warmup_steps = (num_train_steps - 1) // 2
    # num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    # num_train_steps  = max(2, math.ceil((len_train_data - 100) / 500)) * epochs

    # num_train_steps = 10
    # num_warmup_steps = 1

    # single_epoch_steps = max(2, math.ceil((len_train_data - 100) / 500))   # 4
    # num_train_steps    = single_epoch_steps * epochs                        # 20
    # num_warmup_steps   = (single_epoch_steps - 1) // 2                      # 1, NOT 9
        

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler, num_train_steps, num_warmup_steps

def eval_model(logger,params,model_path,tokenizer,cand_encode_path,valid_dataloader, vaild_mention_map):
    params["mode"] = 'test'
    params["path_to_model"] = f'{model_path}/pytorch_model.bin'
    reranker = load_biencoder(params)

    cand_pool_path = params.get("cand_pool_path", None)
    candidate_pool = data.load_or_generate_candidate_pool(
        tokenizer,
        params,
        logger,
        cand_pool_path,
    ) 
    candidate_encoding = None
    if cand_encode_path is not None:
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
            if args.only_get_ent_encoding:
                print(f'only_get_ent_encoding : {args.only_get_ent_encoding}')
                print('No need to generate, already has the encoding!')
                exit()
        except Exception as e:
            logger.info(e)
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = data.encode_candidate(
            reranker,
            candidate_pool,
            params["encode_batch_size"],
            silent=params["silent"],
            logger=logger,
            is_zeshel=params.get("zeshel", None)
            
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)
        
        print(f'args.only_get_ent_encoding {args.only_get_ent_encoding}')
        # input('args.only_get_ent_encoding')

        if args.only_get_ent_encoding:
            exit()  # TODO only for generating candidate encoding
            print(f'only_get_ent_encoding : {args.only_get_ent_encoding}')  

    with open(f'{params["kb_file_path"]}') as f:
        exact_kb = json.load(f)
        if params["onto"] == 'ncbi':
            kb = {}
            for e in exact_kb:
                ent = exact_kb[e]
                kb[e] = ent
                for i in ent['altdiseaseid']:
                    kb[i] = ent
        elif params["onto"] == 'bc5cdr':
            kb = deepcopy(exact_kb)
        elif params["onto"] in ['cmo', 'vt', 'lpt']:
            kb = deepcopy(exact_kb)


    with open(f'{params["data_path"]}/id_map.json') as f:
        map_dict = json.load(f)
        swapped_map_dict = {v: k for k, v in map_dict.items()}

    with open(f'{params["grag_data_path"]}/{params["mode"]}_grag.json') as f:
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
            
    save_results = params.get("save_topk_result")
    new_data, pred_rakings = nnquery.get_topk_predictions(
        reranker,
        valid_dataloader,
        candidate_pool,
        candidate_encoding,
        vaild_mention_map,
        grag_data,
        map_dict,
        exact_kb,
        params,
        params["silent"],
        logger,
        params["top_k"],
        params.get("zeshel", None),
        save_results,
    )

    if save_results: 
        save_data_dir = os.path.join(
            model_path,
            "top%d_candidates" % params['top_k'],
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % params['mode'])
        torch.save(new_data, save_data_path)

        bi_pred_file = save_data_path.replace('.t7', '.json')
        with open(bi_pred_file, 'w') as f:
            json.dump(pred_rakings, f, indent=1)
        
        with open(bi_pred_file) as f:
            bi_all_predictions = json.load(f)
        
        data_dir = params["data_path"]
        kb_dict = {}
        with io.open(f'{data_dir}/kb.jsonl', mode="r", encoding="utf-8") as file:
            for line in file:
                e = json.loads(line.strip())
                kb_dict[e['id']]=e
        with open(f'{data_dir}/id_map.json', 'r') as f:
            id_map = json.load(f)
        with open(f'{params["grag_data_path"]}/{params["mode"]}_grag.json') as f:
            multiple_gt_grag = json.load(f)
            multiple_gt_grag_dict = {}
            for item in multiple_gt_grag:
                multiple_gt_grag_dict[item['sample_id']] = item
        converted = []
        for bi_pred in bi_all_predictions:
            bi_pred_sample_id = bi_pred['mention_data']['sample_id']
            retriever_predictions = []
            for c in bi_pred['retriever_predictions']:
                c['title'] = kb_dict[c['id']]['title']
                c['id'] = id_map[str(c['id'])]
                retriever_predictions.append(c)

            m = bi_pred['mention_data']
            mention = m['mention']
            context_left = m['context_left'].strip()
            context_right = m['context_right'].strip()
            if context_left == '' and context_right != '':
                mention_context = '[MENTION_START] '+mention.strip()+' [MENTION_END] '+context_right
            elif context_left != '' and context_right == '':
                mention_context = context_left+' [MENTION_START] '+mention.strip()+' [MENTION_END]'
            elif context_left == '' and context_right == '':
                mention_context = '[MENTION_START] '+mention.strip()+' [MENTION_END]'
            else:
                mention_context = m['context_left'].strip() + ' [MENTION_START] '+mention.strip()+' [MENTION_END] '+m['context_right'].strip()


            d = {'sample_id' : bi_pred_sample_id,
            'mention': mention,
            'mention_context':mention_context.strip(),
            'ground_truth': multiple_gt_grag_dict[bi_pred_sample_id]['ground_truth'],
            'retriever_result_gt':bi_pred['retriever_retrived_gt'],
            'retriever_predictions':retriever_predictions
            }
            converted.append(d)

        
        with open(f'{params["kb_file_path"]}') as f:
            exact_kb = json.load(f)
        eval_bi = MultiGTEvaluation(converted, exact_kb, 'retriever_predictions', multiple_gt_grag, for_retrieval=True)
        not_none_data, none_data = eval_bi.get_report()
        report = f'Bi-Encoder\n{"_"*20}\n{eval_bi.text_report}\n{"_"*20}'
        with open(f"{bi_pred_file.replace('.json', '_eval.txt')}", 'w') as f:
            f.write(report)

    results = {'normalized_accuracy':eval_bi.recall_at_64}
    return results

def plot_training_curves(per_step_log, diags, output_path):
    
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10

    steps     = [e["step"] for e in per_step_log]
    loss      = [e["loss"] for e in per_step_log]
    lr        = [e["lr"] for e in per_step_log]
    avg_pos   = [e["avg_pos_per_sample"]   for e in per_step_log]
    avg_neg   = [e["avg_neg_per_sample"]   for e in per_step_log]
    avg_gold  = [e["avg_gold_diag_score"]  for e in per_step_log]

    epochs = [d["epoch"] for d in diags]
    accs   = [round((d["accuracy"] * 100), 2) for d in diags]
    gold_score = [d["avg_gold_diag_score"] for d in diags]

    fig, axes = plt.subplots(4, 1, figsize=(12, 13))

    # Row 1: loss + gold diag score
    axes[0].plot(steps, loss,     linewidth=0.8, label="loss")
    axes[0].plot(steps, avg_gold, linewidth=0.8, label="avg_gold_diag_score")
    axes[0].set_ylabel("loss / score")
    axes[0].set_title("Per-step training diagnostics")
    axes[0].legend(loc="upper right", fontsize=8)

    # Row 2: pos / neg counts
    axes[1].plot(steps, avg_pos, linewidth=0.8, label="avg_pos_per_sample")
    axes[1].plot(steps, avg_neg, linewidth=0.8, label="avg_neg_per_sample")
    axes[1].set_ylabel("count")
    axes[1].legend(loc="upper right", fontsize=8)

    # Row 3: learning rate
    axes[2].plot(steps, lr, linewidth=0.8, label="lr", color="tab:gray")
    axes[2].set_ylabel("lr")
    axes[2].set_xlabel("Step")
    axes[2].legend(loc="upper right", fontsize=8)

    # Row 4: epoch vs accuracy
    axes[3].plot(epochs, accs, marker="o", label="acc")
    axes[3].plot(epochs, gold_score, marker="*", label="gold_score")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Acc")
    axes[3].set_title("Epoch vs accuracy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def get_num_epochs(onto):
    if onto.lower() == 'bc5cdr':
        return 4
    else:
        return 1

def main(params):

    # params["num_train_epochs"] = get_num_epochs(params["onto"])
    

    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # rasel : loaded saved model
    prev_path_to_model = deepcopy(params["path_to_model"])
    params["path_to_model"] = params["blink_base_model_path"]
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    params["path_to_model"] = prev_path_to_model

    reranker.set_onto_encoding()


    # # rasel : Freeze bottom N layers to avoid forgetting
    # base_model = model.module if hasattr(model, "module") else model
    # # for idx, (name, _) in enumerate(base_model.named_parameters()):
    # #     print(name)
    #     # if idx == 20:
    #     #     break
    # n_layer = 6  # freeze layers 0, 1, 2
    # for name, param in base_model.named_parameters():
    #     # freeze embeddings of both encoders
    #     if name.startswith("context_encoder.bert_model.embeddings.") or \
    #     name.startswith("cand_encoder.bert_model.embeddings."):
    #         param.requires_grad = False
    #     # freeze bottom N transformer layers of both encoders
    #     if any(name.startswith(f"context_encoder.bert_model.encoder.layer.{i}.") for i in range(n_layer)) or \
    #     any(name.startswith(f"cand_encoder.bert_model.encoder.layer.{i}.") for i in range(n_layer)):
    #         param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print("Frozen:", name)
    # logger.info("\n\nfreeze %d layers.\n\n" % n_layer)
    # # rasel

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    train_samples = utils.read_dataset("train", params["data_path"])
    train_samples = random.sample(train_samples, len(train_samples)) # shuffle in train_samples

    logger.info("Read %d train samples." % len(train_samples))
    
    reranker.set_train_samples(train_samples)
    if len(train_samples) < params["train_batch_size"]:
        new_bs = 128
        logger.info(f"Sample size was smaller than batch size, so new batch size is {new_bs}")
        params["train_batch_size"] = new_bs
    


    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    logger.info("Train batch size : %d " % params["train_batch_size"])
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    train_data, train_tensor_data, train_mention_map = data.process_mention_data(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )

    # data is shuffled in train_samples, no need to to it again
    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    # TODO: reduce duplicated code here
    # valid_samples = utils.read_dataset("valid", params["data_path"])
    valid_samples = utils.read_dataset("test", params["test_data_path"])
    logger.info("Read %d valid samples." % len(valid_samples))
    
     # rasel : 
    removed = False
    for s in valid_samples:
        if 'onto' in s:
            s['world'] = s['onto']
            del s['onto']
            removed = True
        # print(s)
        # input('s')
    if removed:
        print(f'There was a key named "onto", whcih is removed and "world" is set as onto')
    # rasel : 

    valid_data, valid_tensor_data, vaild_mention_map = data.process_mention_data(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )
    

    # rasel

    # evaluate before training
    # results = evaluate(
    #     reranker, valid_dataloader, params, device=device, logger=logger,
    # )

    cand_encode_path = params.get("cand_encode_path", None)
    if cand_encode_path is not None:
        try:
            logger.info("Loading pre-generated candidate encode path.")
            all_candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading all_candidate_encoding failed.")

    logger.info("Evaluation before training!")
    # results = evaluate_with_all_candidate(
    #     reranker, valid_dataloader, all_candidate_encoding, params, device=device, logger=logger,
    # )

    # rasel


    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )



    optimizer = get_optimizer(model, params)
    scheduler, num_train_steps, num_warmup_steps = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    model.train()

    best_epoch_idx = -1
    best_score = -1
    
    per_step_log = []
    global_step = 0

    diags = []
    num_train_epochs = params["num_train_epochs"]
    chunk_size = len(train_dataloader.dataset) / num_train_steps  # float, samples per step
    logger.info(f'chunk_size : {chunk_size}')
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        samples_since_step = 0
        samples_seen = 0
        steps_this_epoch = 0
        next_boundary = chunk_size
        last_batch_idx = len(train_dataloader) - 1

        logger.info(f"\n______________________________ EPOCH {epoch_idx} __________________________________\n")
        epoch_diag = {"avg_pos_per_sample":[], "avg_neg_per_sample":[], "avg_dupes_per_sample": [], "max_dupes_per_sample": [],
               "samples_with_dupes": [], "avg_gold_diag_score": [],
               "diagonal_is_gold_pct": [],"n_unique_golds":[], 'n_kb_neg_sampled':[],'n_total_cands':[],
               "total_neg_count": [],"total_pos_count":[], 'total_dupe_count':[],'total_gold_score':[],
               "total_gold_score": [],"total_rows":[],
               'sample_details':[]}

        tr_loss = 0
        results = None
        epoch_output_folder_path = os.path.join(
                model_output_path, "epoch_{}".format(epoch_idx)
            )

        if not params["only_test_each_epoch"]:
            if params["silent"]:
                iter_ = train_dataloader
            else:
                iter_ = tqdm(train_dataloader, desc="Batch")

            step_count = 0
            for step, batch in enumerate(iter_):
                batch = tuple(t.to(device) for t in batch)
                if len(batch) == 4:
                    context_input, candidate_input, label_ids, sample_ids = batch
                else:
                    context_input, candidate_input, srcs, label_ids, sample_ids = batch

                bs = context_input.size(0)

                loss, _, diag = reranker(context_input, candidate_input, gt_kb_id=label_ids, sample_ids=sample_ids)

                for k, v in diag.items():
                    epoch_diag[k].append(v)

                # if n_gpu > 1:
                #     loss = loss.mean() # mean() to average on multi-gpu.

                # samples_since_step += bs
                # samples_seen += bs
                # (loss * bs).backward()
                # is_last = (step == last_batch_idx)
                # tr_loss += loss.item()
                # step_loss = loss.item()
                # should_step = (
                #     (samples_seen >= next_boundary and steps_this_epoch < num_train_steps)
                #     or is_last
                # )

                if grad_acc_steps > 1:
                    loss = loss / grad_acc_steps
                loss.backward()
                tr_loss += loss.item()
                step_loss = loss.item()
                should_step = ((step + 1) % grad_acc_steps == 0)
      
                # input(f'should_step: \n {should_step} \n')

                if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                    logger.info(
                        "Step {} - epoch {} average loss: {}\n".format(
                            step,
                            epoch_idx,
                            tr_loss / (params["print_interval"] * grad_acc_steps),
                        )
                    )
                    tr_loss = 0

                # if (step + 1) % grad_acc_steps == 0:
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params["max_grad_norm"]
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    logger.info(f"step_count : {step_count}")
                    step_count+=1
                    steps_this_epoch += 1
                    global_step += 1
                    next_boundary = (steps_this_epoch + 1) * chunk_size
                    samples_since_step = 0

                    s_total_rows = diag["total_rows"]
                    per_step_log.append({
                        "step": global_step,
                        "epoch": epoch_idx,
                        "lr": scheduler.get_last_lr()[0],
                        "loss": step_loss,
                        "avg_pos_per_sample": diag["total_pos_count"]   / s_total_rows,
                        "avg_neg_per_sample": diag["total_neg_count"]   / s_total_rows,
                        "avg_dupes_per_sample": diag["total_dupe_count"]   / s_total_rows,
                        "avg_gold_diag_score": diag["total_gold_score"]   / s_total_rows,
                    })
                else:
                    logger.info(f"step + 1 : {step + 1}, grad_acc_steps : {grad_acc_steps}")
                    logger.info(f"(step + 1) % grad_acc_steps : {(step + 1) % grad_acc_steps}")
                    logger.info(f"Step : {step_count}, Epoch : {epoch_idx} has no optimizer update!")

                if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                    logger.info("Evaluation on the development dataset")
                    # evaluate(
                    #     reranker, valid_dataloader, params, device=device, logger=logger,
                    # )
                    # rasel
                    evaluate_with_all_candidate(
                        reranker, valid_dataloader, all_candidate_encoding, params, device=device, logger=logger,
                    )
                    # rasel
                    model.train()
                    logger.info("\n")

                

            logger.info("***** Saving fine - tuned model *****")

            utils.save_model(model, tokenizer, epoch_output_folder_path)

            output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")

        # results = evaluate(
        #     reranker, valid_dataloader, params, device=device, logger=logger,
        # )
        # rasel
        # results = evaluate_with_all_candidate(
        #             reranker, valid_dataloader, all_candidate_encoding, params, device=device, logger=logger,
        #         )
        num_steps = step_count
        total_rows = sum(epoch_diag["total_rows"])
        epoch_summary = {
            "epoch": epoch_idx,
            # row-weighted (honest) averages
            "avg_pos_per_sample":   sum(epoch_diag["total_pos_count"])   / total_rows,
            "avg_neg_per_sample":   sum(epoch_diag["total_neg_count"])   / total_rows,
            "avg_dupes_per_sample": sum(epoch_diag["total_dupe_count"])  / total_rows,
            "max_dupes_per_sample": max(epoch_diag["max_dupes_per_sample"]),
            "pct_samples_with_dupes": sum(epoch_diag["samples_with_dupes"]) / total_rows,
            "avg_gold_diag_score":  sum(epoch_diag["total_gold_score"])  / total_rows,
            # batch-level metadata (these should stay step-averaged since they're per-batch quantities)
            "n_unique_golds":   sum(epoch_diag["n_unique_golds"])   / len(epoch_diag["n_unique_golds"]),
            "n_kb_neg_sampled": sum(epoch_diag["n_kb_neg_sampled"]) / len(epoch_diag["n_kb_neg_sampled"]),
            "n_total_cands":    sum(epoch_diag["n_total_cands"])    / len(epoch_diag["n_total_cands"]),
            "train_batch_size":   params["train_batch_size"],
            "num_steps_in_this_epoch": num_steps,
            "total_warmup_steps": num_warmup_steps,
            "total_rows_processed": total_rows,
            "sample_details": epoch_diag["sample_details"],
        }
        results = eval_model(
            logger,
            params,
            epoch_output_folder_path,
            tokenizer,
            cand_encode_path,
            valid_dataloader,
            vaild_mention_map)
        epoch_summary["accuracy"] = results["normalized_accuracy"]

        diags.append(epoch_summary)
        # rasel

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]


        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = f'{model_output_path}/epoch_{best_epoch_idx}/pytorch_model.bin'
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    with open(f'{model_output_path}/per_step_log.json', "w") as f:
        json.dump(per_step_log, f, indent=2)

    with open(f'{model_output_path}/diagnostics.json', "w") as f:
       json.dump(diags, f, indent=2)

    plot_training_curves(
        per_step_log,
        diags,
        os.path.join(model_output_path, "training_curves.png"),
    )

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)




if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
