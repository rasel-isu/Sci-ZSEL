import wandb
import argparse
import statistics
import numpy as np
import transformers
from transformers import BertTokenizer, \
    get_linear_schedule_with_warmup,get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule, RobertaTokenizer
from disambiguation import *
from data_disambiguation import *
from eval import cat_eval
from utils import *
from datetime import datetime
from torch.optim import AdamW
from tqdm import tqdm
from peft import get_peft_model, LoraConfig



def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds



def load_model(is_init, device, type_loss, args):
    model = ExtractInfoEncoder(args.transformer_model, device, args)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION")
    
    layers = []
    for name, param in model.named_parameters():
        layers.append(name)


    if not is_init:
        print(f"Loading model from pre-trained : {args.model}")

        state_dict = torch.load(args.model) if device.type == 'cuda' else \
                torch.load(args.model, map_location=torch.device('cpu'))


        if args.lora:
            model.model = get_peft_model(model.model, lora_config)
            model.load_state_dict(state_dict['sd'])
        else:
            model.load_state_dict(state_dict['sd'], strict=True)

    if args.fine_tune:
        state_dict = torch.load(args.saved_pt_model) if device.type == 'cuda' else \
            torch.load(args.saved_pt_model, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['sd'])
        if args.lora:
            
            model.model = get_peft_model(model.model, lora_config)
            model = model.to(device)
            
    return model


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)
    
        
    num_train_steps = int(num_train_examples / args.batch /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

            
    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_as_blink(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)
    
        
    num_train_steps = int(num_train_examples / args.batch /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    batch_size = args.batch
    grad_acc = args.gradient_accumulation_steps
    epochs =  args.epochs
    num_train_steps = int(num_train_examples / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * args.warmup_proportion )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
    )
    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = 0
    scheduler = get_constant_schedule(optimizer)
    return optimizer, scheduler, num_train_steps, num_warmup_steps


def get_hit_scores(indices, labels):
    hit = 0
    nums = len(labels)
    for i in range(nums):
        indice = indices[i]
        label = labels[i]
        hit += any([label[index] for index in indice])
    return hit / nums

def evaluate(model, data_loader, device):
    data_loader = tqdm(data_loader, ncols=80)
    labels = []
    pred_data = []
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            model.eval()
            batch = tuple(t.to(device) if not isinstance(t, dict) else t for t in batch)
            raw_data, text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, mention_pos, label = batch
            
            score = model(text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, mention_pos, label,
                           "val")
            label = label.view(-1)
            indice = score.argmax()
            label_ind = label[indice].item()
            labels.append(label_ind)
            raw_data['linked'] = label_ind
            pred_data.append(raw_data)

    return pred_data, sum(labels) / len(labels), 0

def get_rankings(dev_kb, tokenizer, can_input_ids, score, raw_data, logger):

    can_input_id_list = [i for i in can_input_ids]
    score_list = score.view(-1).tolist()
    sublists = [score_list[i:i+3] for i in range(0, len(score_list), 3)]

    row_max_values = [statistics.mean(sublist) for sublist in sublists]

    max_row_index = row_max_values.index(max(row_max_values))
    candidates_id = raw_data['mention_data']['candidates']
    if len(row_max_values) == len(candidates_id) and len(candidates_id) == len(can_input_id_list):
        candidates_score = []
        for s, c, can_input_id in zip(row_max_values, candidates_id, can_input_id_list):
            decoded_text = tokenizer.decode(can_input_id, skip_special_tokens=False)
            decoded_cleaned_text = decoded_text.split('</s>')[0].split('[info2]')[1]
            decoded_cleaned_text = decoded_cleaned_text.strip()
            can_id = c
            onto_ent = dev_kb[can_id]
            onto_text = f"{onto_ent['title']} {onto_ent['text']}"
            onto_text = remove_punctuation(onto_text) 
            if decoded_cleaned_text == onto_text:
                candidates_score.append({'id':can_id, 'title': onto_ent['title'], 'score':s, 'def': onto_ent['text'],})
            else:
                print(0)
        if len(candidates_score) == len(can_input_id_list):
            candidates_sorted = sorted(candidates_score, key=lambda x: x['score'], reverse=True)
            return candidates_sorted
        else:
            logger.log(f'candidates_score and can_input_id_list are not same!')
            return None
        
        
    else:
        logger.log('size of can_input_ids, row_max_values and candidates_id are not same!')
        return None


def evaluate_and_same_predictions(args, model,tokenizer, data_loader, device, logger):
    with open(args.dev_kb) as f:
        dev_kb = json.load(f)

    data_loader = tqdm(data_loader, ncols=80)
    labels = []
    pred_data = []
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            model.eval()
            batch = tuple(t.to(device) if not isinstance(t, dict) else t for t in batch)
            raw_data, text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, mention_pos, label_list = batch
            
            scores = model(text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, mention_pos, label_list,
                           "val")
            
            for score_ind in range(scores.size(0)):
                score = scores[score_ind]
                indice = score.argmax()
                label = label_list[score_ind].view(-1)
                label_ind = label[indice].item()
                labels.append(label_ind)
                sample = {
                    'text':raw_data['text'][score_ind],
                    'mention_data':{
                        'kb_id':raw_data['mention_data']['kb_id'][score_ind],
                        'candidates':[cand[score_ind] for cand in raw_data['mention_data']['candidates']]
                    }
                    }
                sample['linked'] = label_ind
                sample['sample_id'] = raw_data['sample_id'][score_ind].item()
                can_input = can_input_ids[score_ind]
                sample['retrieved_candidates'] = get_rankings(dev_kb, tokenizer, can_input, score, sample, logger)
                pred_data.append(sample)

    return pred_data, sum(labels) / len(labels), 0


def evaluate_test_as_val(epoch, model, tokenizer, device, args, logger):
    dev_data = load_data(args.dev_data)
    dev_entities = load_entities(args.dev_kb)
    args.batch = 32
    dev_data_loader = get_attention_mention_loader(dev_data, dev_entities, tokenizer, False, True, args)
    val_pred, val_acc, _ = evaluate_and_same_predictions(args, model, tokenizer, dev_data_loader, device, logger)
    pred_dir = args.out_dir + f"/{epoch}/"
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = pred_dir + 'pred.json'
    with open(pred_file, 'w') as f: 
        json.dump(val_pred, f, indent=1)
    cat_eval(args, val_acc, pred_file, kbpath=args.dev_kb)
    logger.log(f"test acc: {val_acc}")
    return val_acc

def evaluate_test(model, tokenizer, device, args, logger):
    pred_file = args.pred_data
    forgot_test_data = load_data(args.forgot_test_data)
    forgot_entities = load_entities(args.forgot_kb)
    args.batch = 32 
    forgot_data_loader = get_attention_mention_loader(forgot_test_data, forgot_entities, tokenizer, False, True, args)
    forgot_pred, forgot_acc, _ = evaluate_and_same_predictions(args, model, tokenizer, forgot_data_loader, device, logger)
    with open(pred_file, 'w') as f: 
        json.dump(forgot_pred, f, indent=1)
    forgot_acc = ''
    cat_eval(args, forgot_acc, pred_file, kbpath=args.forgot_kb)
    logger.log(f"test acc: {forgot_acc}")

def train(samples_train, samples_dev, args):
    if args.do_train:
        model_dir =  args.out_dir
        os.makedirs(model_dir, exist_ok=True) 
        hist_file = f'{model_dir}/prime_history.json'
        with open(hist_file, 'w') as f:
            json.dump({}, f, indent=1)
    elif args.do_eval:
        model_dir =  args.out_dir
        os.makedirs(model_dir, exist_ok=True) 

    set_seeds(args)
    best_val_perf = float('-inf')
    logger = Logger(f'{args.out_dir}/train.log', on=True)
    logger.log(str(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.log(f'Using device: {str(device)}', force=True)

    tokenizer = RobertaTokenizer.from_pretrained(args.transformer_model)
    special_tokens = ["[E1]", "[\E1]", '[text]', "[NIL]"]
    sel_tokens = [f"[info{i}]" for i in range(args.info_token_num)]
    special_tokens += sel_tokens
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.tokenizer = tokenizer

    model = load_model(True, device, args.type_loss, args)

    num_train_samples = len(samples_train)
    logger.log(f'number of train samples : {num_train_samples}')
    logger.log(f'number of dev samples : {len(samples_dev)}')
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_as_blink(args, model, num_train_samples)

    args.n_gpu = torch.cuda.device_count()
    model.to(device)
    dp = args.n_gpu > 1

    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)

    train_entities = load_entities(args.train_kb)
    logger.log('number of train entities {:d}'.format(len(train_entities)))
    dev_entities = load_entities(args.dev_kb)
    logger.log('number of dev entities {:d}'.format(len(dev_entities)))
    
    train_loader = get_attention_mention_loader(samples_train, train_entities, tokenizer, True, False, args)
    dev_loader = get_attention_mention_loader(samples_dev, dev_entities, tokenizer, False, True, args, is_dev=True)

    effective_bsz = args.batch * args.gradient_accumulation_steps
    logger.log('***** train *****')
    logger.log('# train samples: {:d}'.format(num_train_samples))
    logger.log('# epochs: {:d}'.format(args.epochs))
    logger.log(' batch size : {:d}'.format(args.batch))
    logger.log(' gradient accumulation steps {:d}'
               ''.format(args.gradient_accumulation_steps))
    logger.log(
        ' effective training batch size with accumulation: {:d}'
        ''.format(effective_bsz))
    logger.log(' # training steps: {:d}'.format(num_train_steps))
    logger.log(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.log(' learning rate: {:g}'.format(args.lr))
    logger.log(' # parameters: {:d}'.format(count_parameters(model)))
    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    start_epoch = 1
    
    ema = EMA(model)

    model.zero_grad()
    if args.do_train:
        wandb.init(project="train_res", 
            name=f'{model_dir}', resume="allow")

        acc_list = []
        
        # logger.log('Before training')
        # hit1 = evaluate_test_as_val(0, model, tokenizer, device, args, logger)
        hit1=0.0
        acc_list.append(round(hit1, 2))
        
        wandb.log({
            "test_accuracy": hit1,
            "epoch": 0})

        for epoch in range(start_epoch, args.epochs + 1):
            logger.log('\nEpoch {:d}'.format(epoch))
            epoch_start_time = datetime.now()
            epoch_train_start_time = datetime.now()

            epoch_dir = args.out_dir + f"/{epoch}/"

            if args.do_eval_only_each_epoch:
                model = ExtractInfoEncoder(args.transformer_model, device, args)
                epoch_model = epoch_dir + f'model.pt'
                state_dict = torch.load(epoch_model) if device.type == 'cuda' else \
                    torch.load(epoch_model, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict['sd'])
                model.to(device)
                dp = args.n_gpu > 1
                if dp:
                    logger.log('Data parallel across {:d} GPUs {:s}'
                            ''.format(len(args.gpus.split(',')), args.gpus))
                    model = nn.DataParallel(model)
                hit1 = evaluate_test_as_val(epoch, model, tokenizer, device, args, logger)
                acc_list.append(round(hit1, 2))
            else:

                train_loader = tqdm(train_loader)
                for step, batch in enumerate(train_loader):
                    model.train()
                    bsz = batch[0].size(0)
                    batch = tuple(t.to(device) for t in batch)
                    count_prime_unable, count_prime, text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, pos, labels = batch

                    loss = model(text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, pos, labels,
                                    "train")

                    if dp:
                        loss = loss.sum() / bsz
                    else:
                        loss /= bsz
                    loss_avg = loss / args.gradient_accumulation_steps

                    loss_avg.backward()
                    tr_loss += loss_avg.item()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    args.clip)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        ema.update(model)        # ema
                        step_num += 1
                        
                logger.log('training time for epoch {:3d} '
                        'is {:s}'.format(epoch, strtime(epoch_train_start_time)))

                ema.apply(model) # Use EMA

                hit1 = evaluate_test_as_val(epoch, model, tokenizer, device, args, logger)
                wandb.log({
                "test_accuracy": hit1,
                "epoch": epoch})
                # hit1 = 0
                acc_list.append(round(hit1, 2))
                try:
                    tloss= tr_loss / step_num
                except ZeroDivisionError:
                    tloss= tr_loss

                logger.log('Done with epoch {:3d} | train loss {:8.4f} | '
                        'recall@1 {:8.4f}|'
                        'recall@5 {:8.4f}'
                        ' epoch time {} '.format(
                    epoch, tloss, hit1, 0,
                    strtime(epoch_start_time)))
                save_model = (hit1 >= best_val_perf)
                torch.save({'opt': args,
                                'sd': model.module.state_dict() if dp else model.state_dict(),
                                'perf': best_val_perf, 'epoch': epoch,
                                'opt_sd': optimizer.state_dict(),
                                'scheduler_sd': scheduler.state_dict(),
                                'tr_loss': tr_loss, 'step_num': step_num,
                                'logging_loss': logging_loss},
                                epoch_dir + f'model.pt')
                
                ema.restore(model)       # training continues from live weights
        
                if save_model:
                    current_best = hit1
                    logger.log('------- new best val perf: {:g} --> {:g} '
                            ''.format(best_val_perf, current_best))

                    best_val_perf = current_best

                count_prime = count_prime.tolist()[-1]
                count_prime_unable= count_prime_unable.tolist()[-1]
                
            fname = 'without_test_title'
            if args.use_title_during_testing:
                fname = 'with_test_title'

            img_ile = ''
            plot_acc(model_dir, img_ile, 
                    [i for i in range(len(acc_list))] , acc_list)
            
            if not args.do_eval_only_each_epoch:
                logger.log(f'prime ent added {count_prime} times!')
                logger.log(f'No. of epoch was {args.epochs} so, {count_prime/args.epochs} prime ent used in training!')
                logger.log(f'No. of epoch was {args.epochs} so, {count_prime_unable/args.epochs} prime ent not found during training!')

    if args.do_eval:
        model = load_model(False, device, args.type_loss, args).to(device)
        evaluate_test(model, tokenizer, device, args, logger)

def main(args):
    train_data = load_data(args.train_data)
    dev_data = load_data(args.dev_data)
    train(train_data, dev_data, args)
