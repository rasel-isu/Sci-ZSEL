import os
import re
import torch
import json
import random
import time
import numpy as np
from blink.crossencoder.eval_utils import evaluate_cat_wise
from utils import plot_train_test_accuracy, read_jsonl
import wandb
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from blink.crossencoder.crossencoder import CrossEncoderRanker, EMA
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS
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

def modify_list(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = [i.tolist() for i in candidate_input]


    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(torch.LongTensor(mod_input))

    padded_input = pad_sequence(new_input, batch_first=True, padding_value=-1)
    lengths = torch.LongTensor([len(item) for item in new_input])

    

    # input('Stop : ')
    return padded_input, lengths


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

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
    )

    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def get_scheduler_custom(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
    )
    
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler

def remove_special_chars(text):
    text = re.sub(r'[^A-Za-z0-9]', '', text)
    return text


def main(params):
    wandb.init(dir="wandb_logs", project="train_cross", 
               name=f'{params["output_path"]}-context-{params["max_context_length"]}-batch-{params["train_batch_size"]}', resume="allow")
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])
    logger.info(json.dumps(params, indent=1))

    # Init model
    prev_path = params["path_to_model"]
    params["path_to_model"] = params["blink_model_path"]
    reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    params["path_to_model"] = prev_path

    device = reranker.device
    n_gpu = reranker.n_gpu


    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
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

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]
    
    fname = os.path.join(params["data_path"], "train.t7")
    train_data = torch.load(fname)
    context_input = train_data["context_vecs"]
    label_input = train_data["labels"]
    num_train_sample = len(label_input)


    candidate_input = train_data["candidate_vecs"]
    sample_ids = train_data["sample_ids"]



    if  params["exclude_gt"]:
        data_path = params["raw_data_path"][:-1]
        train_sample_pseudo = [i['label_id'] for i in read_jsonl(f'{data_path}/train.jsonl')]
        kb_integer_ids = train_data["candidate_kb_integer_ids"]
        train_sample_with_gt = [i['label_id'] for i in read_jsonl(f'{data_path}_GT/train.jsonl')]
        kb_integer_ids_text = ''
        label_input_gt = []
        for smpl, lbl_gt in zip(kb_integer_ids.tolist(), train_sample_with_gt):
            lbl_gt_idx = -1
            for i in range(len(smpl)):
                if smpl[i] == lbl_gt:
                    lbl_gt_idx = i
                kb_integer_ids_text+=f'{i}-->{smpl[i]}, '
            label_input_gt.append(lbl_gt_idx)
            kb_integer_ids_text+=f'\n'

        label_input_gt = torch.tensor(label_input_gt)

    connected_labels = train_data["connected_labels_graph"]
    connected_candidates_graph = train_data["connected_candidates_graph"]
    connected_candidates, connected_candidate_len = modify_list(context_input, connected_candidates_graph, max_seq_length)
    conn_neg_len = [item-1 for item in connected_candidate_len.tolist()]
    logger.info(f"Min prch of label : {min(conn_neg_len)}")
    logger.info(f"Max prch of label : {max(conn_neg_len)}")
    logger.info(f"Average prch of label : {round(sum(conn_neg_len) / len(conn_neg_len),2)}")

    context_input = modify(context_input, candidate_input, max_seq_length)
    
    if params["zeshel"]:
        src_input = train_data['worlds'][:len(context_input)]
        if params["exclude_gt"]:
            train_tensor_data = TensorDataset(context_input, label_input, src_input, label_input_gt)
        else:
            train_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        if params["exclude_gt"]:
            train_tensor_data = TensorDataset(context_input, label_input, label_input_gt)
        else:
            # train_tensor_data = TensorDataset(context_input, label_input, sample_ids)
            train_tensor_data = TensorDataset(context_input, label_input, sample_ids,
                                connected_candidates, connected_candidate_len, connected_labels)

    train_sampler = RandomSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, 
        sampler=train_sampler, 
        batch_size=params["train_batch_size"]
    )

    test_set_name = 'test'
    fname = os.path.join(params["data_path"], f"{test_set_name}.t7")
    if params['test_data_path'] != '':
        fname = os.path.join(params["test_data_path"], f"{test_set_name}.t7")
    valid_data = torch.load(fname)
    context_input = valid_data["context_vecs"]
    candidate_input = valid_data["candidate_vecs"]
    label_input = valid_data["labels"]
    candidate_kb_integer_ids = valid_data["candidate_kb_integer_ids"]
    sample_ids = valid_data["sample_ids"]

    context_input = modify(context_input, candidate_input, max_seq_length)
    if params["zeshel"]:
        src_input = valid_data["worlds"][:len(context_input)]
        valid_tensor_data = TensorDataset(context_input, label_input, src_input)
    else:
        valid_tensor_data = TensorDataset(context_input, label_input, candidate_kb_integer_ids, sample_ids)

    valid_sampler = SequentialSampler(valid_tensor_data)

    valid_dataloader = DataLoader(
        valid_tensor_data, 
        sampler=valid_sampler, 
        batch_size=params["eval_batch_size"]
    )

    if params["only_infer_test_set"]:
        results = evaluate_cat_wise(
                params,
                test_set_name,
                model_output_path + '/before_fine_tune',
                reranker,
                valid_dataloader,
                device=device,
                logger=logger,
                context_length=context_length,
                zeshel=params["zeshel"],
                silent=params["silent"],)
        exit()
    
    if params["onto"] == 'bc5cdr':
        results = {'recall_at_1':0.0}
    elif params["onto"] == 'ncbi_disease':
        results = {'recall_at_1':0.0}
    elif params["onto"] == 'cmo':
        results = {'recall_at_1':0.0}
    elif params["onto"] == 'vt':
        results = {'recall_at_1':0.0}
    elif params["onto"] == 'lpt':
        results = {'recall_at_1':0.0}
    elif params["onto"] == 'cometa':
        results = {'recall_at_1':0.0}
    wandb.log({ "test_accuracy": results['recall_at_1'],"epoch": 0})
    time_start = time.time()
    utils.write_to_file(os.path.join(model_output_path, "training_params.txt"), str(params))

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler_custom(params, optimizer, len(train_tensor_data), logger)

    ema = EMA(model)
    model.train()
    best_epoch_idx = -1
    best_score = -1
    num_train_epochs = params["num_train_epochs"]
    torch.cuda.empty_cache()

    acc_and_epoch = []
    if params["silent"]:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader, desc="Batch")

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        
        if params["only_test_each_epoch"]:
            epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx))
            params["path_to_model"] = f'{epoch_output_folder_path}/pytorch_model.bin'
            reranker = CrossEncoderRanker(params)
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

            continue

        tr_loss = 0
        results = None
        part = 0
        train_correct_count = 0
        reranker.prch_added_as_neg = []
        reranker.epoch = epoch_idx
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0] 
            label_input = batch[1]
            sample_id = batch[2]
            graph_candidates = [batch[3], batch[4], batch[5]]
            if params["exclude_gt"]:
                label_input_gt = batch[2]
                loss, scores = reranker(context_input, label_input, context_length, label_input_gt)
            else:
                loss, scores, correct_count = reranker(context_input, label_input, context_length, 
                    graph_candidates=graph_candidates, sample_id=sample_id, is_train=True)

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps
            
            tr_loss += loss.item()
            train_correct_count+=correct_count
            wandb.log({"train_loss": loss.item()})  

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            if params['cross_enc_negative_selection'] not in [
                'prnt_chld_as_pos',
                 ]:
                loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ema.update(model)        # ema


            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                logger.info("***** Saving fine - tuned model *****")
                epoch_output_folder_path = os.path.join(
                    model_output_path, "epoch_{}_{}".format(epoch_idx, part)
                )
                part += 1
                utils.save_model(model, tokenizer, epoch_output_folder_path)
                model.train()
                logger.info("\n")

        logger.info(f'error_count  : {reranker.error_count }')
        prch_added_as_neg = reranker.prch_added_as_neg
        if prch_added_as_neg:
            logger.info(f"Min prch at epoch {epoch_idx} : {min(prch_added_as_neg)}")
            logger.info(f"Max prch at epoch {epoch_idx} : {max(prch_added_as_neg)}")
            logger.info(f"Average prch at epoch {epoch_idx} : {round(sum(prch_added_as_neg)/len(prch_added_as_neg),2)}")
            
        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx))
        

        ema.apply(model) # Use EMA
        utils.save_model(model, tokenizer, epoch_output_folder_path)
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
        ema.restore(model)       # so training continues from live weights

        logger.info(f"epoch : {epoch_idx}, recall@1 : {results['recall_at_1']}")
        wandb.log({
            "test_accuracy": results['recall_at_1'],
            "epoch": epoch_idx+1
        })
        acc_and_epoch.append({
            'train':{'acc':round(train_correct_count/num_train_sample, 2)},
            'test':{'acc':round(results['recall_at_1'], 2)}
        })


        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    plot_train_test_accuracy(acc_and_epoch, f'{model_output_path}/train_hist.png')
    with open(f'{model_output_path}/train_hist.json', 'w') as f:
        json.dump(acc_and_epoch, f)
    
    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )


if __name__ == "__main__":    
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()
    args = parser.parse_args()
    params = args.__dict__
    main(params)
