from copy import deepcopy
import json
import os
from tqdm import tqdm

from utils import CONFIG, check_duplicate, read_jsonl, make_blink_format, get_category,save_kb_and_id_map,is_accuratly_pseudo_labelled,gt_mention_category_count

def get_prime_kb(onto_path, world, split_name, prime_kb_file=None):
    if prime_kb_file is None:
        prime_kb_file = f'{onto_path}{world}_prime_{split_name}_newly_generated.json'
    
    with open(prime_kb_file, 'r') as f:
        prime_def = json.load(f)
        print(f'prime_def : {len(prime_def)} : {prime_kb_file}')
        kb_prime_def = {}
        for i in prime_def:
            kb_prime_def[i['document_id']] = i

    return kb_prime_def

def grag_to_blink_m1_from_kb(
        source_dir, 
        source_file, 
        kb_file_path,
        title_key, 
        defi_key, 
        world, 
        split_name,
        out_file,
        out_path=None,
        mtokens=['[MENTION_START]', '[MENTION_END]'],
        plurals=False,
        original_title=True,
        only_correct_label=False,
        has_gt=True
        ):

    with open(source_dir+source_file, 'r') as f:
        corpus = json.load(f)
    print(f'corpus : {len(corpus)} : {source_dir+source_file}')

    with open(kb_file_path, 'r') as f:
        kb = json.load(f)
        kb_name_dict = {}
        for i in kb:
            name = kb[i]['name']
            kb_name_dict[name.lower().strip()] = deepcopy(kb[i])

    print(f'kb : {len(kb)} : {kb_file_path}')

    if not out_path:
        out_path = f'data/{world}/blink_format/{split_name}/'
    os.makedirs(out_path, exist_ok=True)

    if not original_title:
        prime_kb_file = f'{out_path}{world}_prime_{split_name}_newly_generated.json'
        kb_prime_def = get_prime_kb(out_path, world, split_name, prime_kb_file)

    map_dict = save_kb_and_id_map(world, out_path, kb, title_key, defi_key)

    cat_wise_acc = {
        "HO":{'count':0, 'acc':0, 'correct':0}, 
        "MINT":{'count':0, 'acc':0, 'correct':0},
        "LO":{'count':0, 'acc':0, 'correct':0},
        "NO":{'count':0, 'acc':0, 'correct':0}}
    
    incorrectly_labelled_samples = []
    men_lbl_gt=[]
    plurals_count=0
    count_ho = 0
    c_gt_not_in_kb = 0
    oot_file_path = f'{out_path}{out_file}'
    saved_samples = []
    unique_sample = {}
    with open(oot_file_path, 'w') as f:
        for doc in tqdm(corpus):
            mention = doc['mention'].lower().strip()
            if not plurals:
                if mention in kb_name_dict:
                    pseudo_ent = kb_name_dict[mention]
                    if original_title:
                        label_title = pseudo_ent['name']
                    else:
                        label_title = kb_prime_def[mention]['newly_generated_name']

                    label_def = pseudo_ent['def']
                    label_id = pseudo_ent['id']
                    d = make_blink_format(world, doc, mtokens, kb, map_dict, defi_key, 
                                            label_title, label_def, label_id)
                    if d not in saved_samples:
                        count_ho+=1
                        category = get_category(mention, label_title)
                        cat_wise_acc[category]['count']+=1
                        
                        if has_gt:
                            gt_id_list = [i['id'] for i in doc['ground_truth'] ]
                            is_accurate = is_accuratly_pseudo_labelled(gt_id_list, pseudo_ent)
                            if is_accurate:
                                cat_wise_acc[category]['correct']+=1

                            if only_correct_label:
                                if not is_accurate:
                                    incorrectly_labelled_samples.append(json.dumps(d))
                                    men_lbl_gt.append({
                                        'm':doc['mention'],
                                        'lbl':label_title,
                                        'gt':'|'.join([str(i['title']) for i in doc['ground_truth']])
                                    })
                                    continue


                        
                        smpl_d = deepcopy(d)
                        if 'sample_id' in smpl_d:
                            del smpl_d['sample_id']
                        nk = ''
                        for k in smpl_d:
                            nk+=f'{smpl_d[k]}'
                        if nk not in unique_sample:
                            unique_sample[nk]=0
                            saved_samples.append(d) 
                            f.write(json.dumps(d) + "\n")
            else:
                for ent in kb_name_dict:
                    subcat = check_plural(mention, ent)
                    if subcat == "plural":
                        plurals_count+=1
                        label_title = kb_name_dict[mention]['name']
                        label_def = kb_name_dict[mention]['def']
                        label_id = kb_name_dict[mention]['id']
                        d = make_blink_format(world, doc, mtokens, kb, map_dict, defi_key, 
                                                label_title, label_def, label_id)
                        if d not in saved_samples:
                            saved_samples.append(d) 
                            f.write(json.dumps(d) + "\n")
                            category = get_category(mention, label_title)
                            cat_wise_acc[category]['count']+=1
                            if label_title.lower().strip() == gt_list:
                                cat_wise_acc[category]['correct']+=1
                            

    print(f'{plurals_count} : plurals added')
    check_duplicate(oot_file_path)

    overall = {'count':0, 'acc':0, 'correct':0}
    for cat in cat_wise_acc:
        try:
            overall['count']+=cat_wise_acc[cat]['count']
            overall['correct']+=cat_wise_acc[cat]['correct']
            cat_wise_acc[cat]['acc'] =round((cat_wise_acc[cat]['correct']/cat_wise_acc[cat]['count'])*100,2)
        except ZeroDivisionError:
            pass

    try:
        overall['acc'] = round((overall['correct']/overall['count'])*100,2)
    except ZeroDivisionError:
        pass
    
    cat_wise_acc['OVERALL'] = overall
    
    with open(f'{out_path}cat_wise_acc.json', 'w') as f:
        json.dump(cat_wise_acc, f, indent=2)

    counts = gt_mention_category_count(f"{out_path}", out_file)


    if only_correct_label:
        with open(f'{out_path}incorrectly_labelled_samples.jsonl', 'w') as f:
            f.write('\n'.join(incorrectly_labelled_samples))

        with open(f'{out_path}mention_label_gt.json', 'w') as f:
            json.dump(men_lbl_gt, f, indent=1)


def get_top_k_from_biencoder(split_name,kb_file_path,mc_file, data_from_exp, k=1):
    with open(mc_file, 'r') as f:
        mc = json.load(f)


    data_dir = f'{CONFIG["data_dir"]}/blink_format/{split_name}/{data_from_exp}/'
    # data_dir = f'data/{world}/blink_format/before_medic/{split_name}/{data_from_exp}/'

    with open(f'{data_dir}id_map.json', 'r') as f:
        id_map = json.load(f)
    with open(kb_file_path, 'r') as f:
        kb_dict = json.load(f)
    
    top_candidates = {}
    for i in mc:
        retriever_predictions = i['retriever_predictions'][:k]
        for c in retriever_predictions:
            c['id'] = id_map[str(c['id'])]
            ent = kb_dict[c['id']]
            top_candidates[c['id']]=ent

    c_no_def = 0
    top_candidates_have_def = {}
    for can in top_candidates:
        if top_candidates[can]['def'] !="":
            top_candidates_have_def[can]=top_candidates[can]
        else:
            c_no_def+=1
    print(f'Total mentions : {len(mc)}')
    print(f'Unique entities : {len(top_candidates)}')
    print(f'Unique entities those have def: {len(top_candidates_have_def)}')
    print(f'missed for not having def: {len(top_candidates)-len(top_candidates_have_def)}')

    with open(data_dir+f'{CONFIG["biencoder_top1_file"]}', 'w') as f:
        json.dump(top_candidates_have_def,f, indent=1)

        
    with open(data_dir+f'{CONFIG["biencoder_top1_file"].replace(".json", "_dont_rm_if_no_def.json")}', 'w') as f:
        json.dump(top_candidates,f, indent=1)

def get_unq_ent_from_m1_e1(split_name,kb_file_path, data_from_exp):
    data_dir = f'{CONFIG["data_dir"]}/blink_format/{split_name}/{data_from_exp}/'
    mc = read_jsonl(f'{data_dir}train.jsonl')
    with open(f'{data_dir}id_map.json', 'r') as f:
        id_map = json.load(f)
    with open(kb_file_path, 'r') as f:
        kb_dict = json.load(f)
    
    labels = {}
    for i in mc:
        label_id = id_map[str(i['label_id'])]
        ent = kb_dict[label_id]
        labels[label_id]=ent
    c_no_def = 0
    labels_have_def = {}
    for lbl in labels:
        if labels[lbl]['def'] !="":
            labels_have_def[lbl]=labels[lbl]
        else:
            c_no_def+=1
    print(f'Total mentions : {len(mc)}')
    print(f'Unique entities : {len(labels)}')
    print(f'Unique entities those have def: {len(labels_have_def)}')
    print(f'missed for not having def: {len(labels)-len(labels_have_def)}')
    with open(data_dir+f'{CONFIG["exact_match_file"]}', 'w') as f:
        json.dump(labels_have_def,f, indent=1)

        
    with open(data_dir+f'{CONFIG["exact_match_file"].replace(".json", "_dont_rm_if_no_def.json")}', 'w') as f:
        json.dump(labels,f, indent=1)


def get_ents_if_mention_overlaps_onto_term(kb_file_path, onto_path):
    source_file = onto_path+'train.jsonl'
    corpus = read_jsonl(source_file)
    print(f'corpus : {len(corpus)} : {source_file}')
    with open(kb_file_path, 'r') as f:
        kb = json.load(f)

    ho_m_eql_e_count = 0
    mint_m_eql_e_count = 0
    lo_m_eql_e_count = 0
    men_eql_ent = {}

    for doc in tqdm(corpus):
        mention = doc['mention'].lower().strip()
        for ent in kb:
            ent_name = kb[ent]['name'].lower().strip()
            category = get_category(mention, ent_name)
            if category!='NO':
                kb[ent]['mention'] = doc['mention']
                men_eql_ent[ent]=kb[ent]
                if category=='HO':
                    ho_m_eql_e_count+=1
                elif category=='MINT':
                    mint_m_eql_e_count+=1
                elif category=='LO':
                    lo_m_eql_e_count+=1

    print(f'Total mentions : {len(corpus)}')
    print(f'mention=entity for HO: {ho_m_eql_e_count}')
    print(f'mention=entity for MINT: {mint_m_eql_e_count}')
    print(f'mention=entity for LO: {lo_m_eql_e_count}')
    print(f'Unique entities : {len(men_eql_ent)}')

    c_no_def = 0
    ents_have_def = {}
    for entity in men_eql_ent:
        if men_eql_ent[entity]['def'] !="":
            ents_have_def[entity]=men_eql_ent[entity]
        else:
            c_no_def+=1
            
    print(f'Unique entities those have def: {len(ents_have_def)}')
    print(f'missed for not having def: {len(men_eql_ent)-len(ents_have_def)}')

    with open(onto_path+f'mention_overlaps_onto_term.json', 'w') as f:
        json.dump(ents_have_def,f, indent=1)


def selection(kb_file_path, world, split_name, source_file):
    print(f"{'_'*20}{world}{'_'*20}")
    hast_gt = CONFIG['has_ground_truth']

    # Exact match entities
    out_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/(m1_e1)/"
    # Construction from exact match 
    grag_to_blink_m1_from_kb(source_dir=CONFIG['data_dir'],source_file=source_file, 
            kb_file_path=kb_file_path,title_key='name',defi_key='def',world=world, 
            split_name=split_name,out_file='train.jsonl',out_path=out_path, plurals=False, 
            has_gt=hast_gt # dont evaluate if no GT
            )

    data_from_exp = '(m1_e1)'
    get_unq_ent_from_m1_e1(split_name, kb_file_path, data_from_exp)

    # Bi-encoder top-1 entities
    data_from_exp = 'original_data'
    mc_file = f'{CONFIG["saved_model_dir"]}/biencoder/{split_name}/{data_from_exp}/top64_candidates/train.json'
    get_top_k_from_biencoder(split_name, kb_file_path, mc_file, data_from_exp, k=1)


def main():
    world = CONFIG['world']
    kb_file = CONFIG['kb_file']
    data_dir = CONFIG['data_dir']
    kb_file_path = f'{data_dir}/{kb_file}'
    
    selection(kb_file_path, world, 'train', f'train_grag.json')

if __name__ == "__main__":
    main()
