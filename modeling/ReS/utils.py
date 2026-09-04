import json
import re
import sys
import os
import random
import bisect
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

with open('../../config.json') as f:
    CONFIG = json.load(f)
    
class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()


def print_run_config(logger, exp_settings, all_f_settings, lora):
    logger.log("============================================")
    logger.log("Run configuration:")
    logger.log(f"lora: {lora}")

    logger.log("exp_settings:")
    for setting in exp_settings:
        logger.log(
            f"  corpus={setting['corpus_name']}, "
            f"onto={setting['onto_name']}, "
            f"split={setting['split_name']}, "
            f"use_title={setting['use_title_during_testing']}"
        )

    logger.log("all_f_settings:")
    for f_setting in all_f_settings:
        logger.log(f"  {f_setting}")

    logger.log("============================================")


def sample_range_excluding(n, k, excluding):
    skips = [j - i for i, j in enumerate(sorted(set(excluding)))]
    s = random.sample(range(n - len(skips)), k)
    return [i + bisect.bisect_right(skips, i) for i in s]

def read_data(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(x) for x in f]
    return data

def pad_values(data, token, max_len):
    return (data + [token for _ in range(max_len)])[:max_len]

def remove_punctuation(sentence):
    remove_chars = '[’!"#$%&\'()*+,-.:;<=>?@，。?★、…【】《》？“”‘’！\\^_`{|}~]+'
    result = re.sub(remove_chars, ' ', sentence)
    result = ' '.join(result.split())
    return result

def plot_acc(dir, name, x,y):
    plt.plot(x, y, marker='o')
    plt.title("Accuracy and Epoch on Val set")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if dir[-1] == '/':
        dir = dir[:-1]
    if '/' in name:
        name = name.replace('/', '')
    plt.savefig(f"{dir}/acc_epoch_{name}.png", dpi=300)
    plt.close()

def summary_for_test_ent_appears_in_train(train_file, test_file):
    with open(train_file, 'r') as f:
        train = json.load(f)
    train_ents = {}
    for td in train:
        train_ents[td['mention_data']['kb_id']] = td

    count_appers = 0
    count_didnt_appers = 0

    with open(test_file, 'r') as f:
        test = json.load(f)
    for td in test:
        test_id = td['mention_data']['kb_id']
        if test_id in train_ents:
            count_appers+=1
        else:
            count_didnt_appers+=1


    print(f'train : {len(train)}\ntest : {len(test)}\n{count_appers} GT entities from test set is also appears as GT in train set')

def compare_with_multiple_gt(candidate, gt_list):
    for gt_item in gt_list:
        if gt_item['id'] == candidate['id']:
            return True, gt_item
    return False, {}
    
        
def compare_with_multiple_gt_with_altid(kb, candidate, gt_list):
    candidate_id = candidate['id']
    if candidate_id != '-1':
        if 'altdiseaseid' in kb[candidate_id]:
            altids = kb[candidate_id]['altdiseaseid']
            for alid in altids: 
                for gt_item in gt_list:
                    if gt_item['id'] == alid:
                        return True, gt_item,  alid
                
    return False, {}, None

def check_plural(mention, title):
    if title is None:
        title = ''
    mention, title = mention.lower(), title.lower()
    # if len(mention) != len(gt_title):
    longer, shorter = (mention, title) if len(mention) > len(title) else (title, mention)
    diff = len(longer)-len(shorter)
    if diff ==1 and longer.startswith(shorter) and longer.endswith("s"):
        return 'plural'
    else:
        return 'Pure'
    
def get_category(mention, title):

    mention_lower = mention.lower().replace(',', ' ')
    if title is None:
        title = ''
    title_lower = title.lower().replace(',', ' ')
    subcat = check_plural(mention_lower, title_lower)

    if mention_lower == title_lower:
        return 'HO'
    elif subcat=='plural':
        return 'HO'
    elif mention_lower in title_lower and title_lower != mention_lower:
        return 'MINT'
    else:
        words_mention = set(mention_lower.split())
        words_title = set(title_lower.split())
        common_words = words_mention.intersection(words_title)
        if common_words:
            return "LO"
        else:
            for word in words_mention:
                if f' {word} '  in f' {title_lower} ':
                    return "LO"

            return "NO"



class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}