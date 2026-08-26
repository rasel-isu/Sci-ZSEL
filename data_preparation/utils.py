from copy import deepcopy
import json
import os
import random
import re
import io
from tqdm import tqdm
from sentence_transformers import util

with open('../config.json') as f:
    CONFIG = json.load(f)

def read_jsonl(filename):
    data = []
    if '.jsonl' in filename:
        with io.open(filename, mode="r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line.strip()))
    return data




def check_duplicate(out_file):
    all_data = read_jsonl(out_file)
    unique_sample = {}
    for m in all_data:
        smpl_d = deepcopy(m)
        if 'sample_id' in smpl_d:
            del smpl_d['sample_id']
        nk = ''
        for k in smpl_d:
            nk+=f'{smpl_d[k]}'
        if nk not in unique_sample:
            unique_sample[nk]=0

    print(f'{len(all_data)} : {out_file}')
    print(f'{len(unique_sample)} : unique samples')

    if len(all_data) != len(unique_sample):
        raise ValueError('Some samples are duplicated!')

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
        
def is_accuratly_pseudo_labelled(gt_id_list, pseudo_ent):

    
    if 'altdiseaseid' in pseudo_ent:
        pseudo_label_id_list = [pseudo_ent["id"]] + pseudo_ent["altdiseaseid"]
    else:
        pseudo_label_id_list = [pseudo_ent["id"]]

    for pseudo_label_id in pseudo_label_id_list:
        if pseudo_label_id in gt_id_list:
            return True

def gt_mention_category_count(source_dir, filename):
    data = read_jsonl(source_dir+filename)
    cat_wise_gt_matched = {
            "HO":{'count':0, 'items':[]}, 
            # "plural":{'count':0, 'items':[]}, 
            "MINT":{'count':0, 'items':{
                'plural':{'count':0, 'items':[]},
                'Pure':{'count':0, 'items':[]}} 
            },
            "LO":{'count':0, 'items':{
                'plural':{'count':0, 'items':[]},
                'Pure':{'count':0, 'items':[]}} 
            },
            "NO":{'count':0, 'items':{
                'plural':{'count':0, 'items':[]},
                'Pure':{'count':0, 'items':[]}} 
            }
            }
    unq_mention = {}
    unq_gt = {}
    for i in data:
        try:
            mention = i['mention']
        except Exception as e:
            print(i)
            input('s')
        title = i['label_title']
        if mention in unq_mention:
            unq_mention[mention]+=1
        else:
            unq_mention[mention]=1

        if title in unq_gt:
            unq_gt[title]+=1
        else:
            unq_gt[title]=1

        # if title == None:
        #     title = ''
            
        # if mention == 'autosomal dominant disease':
        #     print(0)
        category = get_category(mention, title)
        # category_old = category_specific_count(mention, title)
        # if category!=category_old:
        #     print(0)
        
        item_d = {
                'mention' : mention,
                'gt_title' : title,
                'gt_id' : i['label_id']
            }
        if category=="HO":
            cat_wise_gt_matched[category]['count']+=1
            cat_wise_gt_matched[category]['items'].append(item_d)
        else:
            cat_wise_gt_matched[category]['count']+=1
            subcat = check_plural(mention, title)
            cat_wise_gt_matched[category]['items'][subcat]['count']+=1
            cat_wise_gt_matched[category]['items'][subcat]['items'].append(item_d)
            
    cat_wise_gt_matched['unique_mention_count'] = len(unq_mention)
    cat_wise_gt_matched['unique_gt_count'] = len(unq_gt)

    sorted_unq_mention = sorted(unq_mention.items(), key=lambda item: item[1], reverse=True)
    cat_wise_gt_matched['unique_mention'] =  dict(sorted_unq_mention)

    sorted_unq_gt = sorted(unq_gt.items(), key=lambda item: item[1], reverse=True)
    cat_wise_gt_matched['unique_gt'] =  dict(sorted_unq_gt)

    with open(source_dir+filename.replace('.jsonl', '_category_count.json'), 'w') as f:
        json.dump(cat_wise_gt_matched, f, indent=1)

    return cat_wise_gt_matched


def fix_entity_catalogue(filename):
    c = 0
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            try:
                ent = json.loads(line)
                if 'title' not in ent:
                    ent['title'] = ent['entity']
                    # input('title : ')
                    print(ent)
                    c+=1
            except Exception as e:
                print(e)
                print(line)
    print(c)
    
def make_blink_format(world, doc, mtokens, 
                          kb, map_dict, defi_key, label_title,
                          label_def=None, label_id=None):
    context_left = doc['mention_context'].split(mtokens[0])[0]
    context_right = doc['mention_context'].split(mtokens[1])[1]
    m = doc['mention']
    if not label_id:
        label_id = doc['ground_truth']['id']
    if label_def:
        defi = label_def
    else:
        defi = kb[label_id][defi_key]

    if label_id in map_dict:
        mapped_label_id = map_dict[label_id]
    else:
        mapped_label_id = -1

    d = {
        "sample_id":doc['sample_id'],
        "context_left": context_left,
        "context_right": context_right,
        "mention":m,
        "label": defi,
        "label_id": mapped_label_id,
        "label_title": label_title,
        "onto": world
        }
    return d

def grag_to_blink_original_def(
        source_dir, 
        source_file, 
        kb_file_path,
        title_key, 
        defi_key, 
        world, 
        split_name,
        out_file,
        out_path,
        skip_sample_if_ent_not_in_kb=False,
        gt_title_key='title',
        mtokens=['[MENTION_START]', '[MENTION_END]'],
        only_ho=True,
        ho_prime_others_not=False,
        sample_size=None,
        original_title=False,
        num_sample_for_theta=None,
        has_gt=True
        ):

    with open(source_dir+source_file, 'r') as f:
        corpus = json.load(f)

        if sample_size:
            random.seed(0)
            corpus = random.sample(corpus, sample_size)

    print(f'corpus : {len(corpus)} : {source_dir+source_file}')

    if CONFIG['has_ent_alt_id']:
        with open(kb_file_path, 'r') as f:
            exact_kb = json.load(f)
            kb = {}
            for e in exact_kb:
                ent = deepcopy(exact_kb[e])
                kb[e] = ent
                for i in ent['altdiseaseid']:
                    kb[i] = ent
            print(f'Num ent in KB : {len(exact_kb)}')
    else:
        with open(kb_file_path, 'r') as f:
            exact_kb = json.load(f)
            kb = deepcopy(exact_kb)

    os.makedirs(out_path, exist_ok=True)

    if only_ho is True or ho_prime_others_not is True:
        prime_kb_file = f'{out_path}{world}_prime_{split_name}_newly_generated.json'
        with open(prime_kb_file, 'r') as f:
            prime_def = json.load(f)
            print(f'prime_def : {len(prime_def)} : {prime_kb_file}')
            kb_prime_def = {}
            for i in prime_def:
                kb_prime_def[i['document_id']] = i

    cat_wise_acc = {
        "HO":{'count':0, 'acc':0, 'correct':0}, 
        "MINT":{'count':0, 'acc':0, 'correct':0},
        "LO":{'count':0, 'acc':0, 'correct':0},
        "NO":{'count':0, 'acc':0, 'correct':0}}



    map_dict = {}
    ent_list = []
    id_map={}
    kb_file = f'{out_path}kb.jsonl'
    with open(kb_file, 'w') as f:
        for i, id in enumerate(exact_kb):
            inc_id = i
            id_map[inc_id]=id
            map_dict[id]=inc_id
            exact_kb[id]['id'] =inc_id
            d = {'id':inc_id, 'title':exact_kb[id][title_key],'text':exact_kb[id][defi_key]}
            json_str = json.dumps(d)
            f.write(json_str + "\n")
            dn = {'idx':f"{world}?curid={inc_id}", 'entity':exact_kb[id][title_key], 'title':exact_kb[id][title_key],'text':exact_kb[id][defi_key]}
            ent_list.append(dn)

    print(f'created kb : {len(ent_list)} : {kb_file}')

    id_map_file=f'{out_path}id_map.json'
    with open(id_map_file, 'w') as f:
        json.dump(id_map,f)
    print(f'created id map  : {len(id_map)} : {id_map_file}')

    
    entity_file = f'{out_path}entity.jsonl'
    with open(entity_file, 'w') as f:
        for ent in ent_list:
            json_str = json.dumps(ent)
            f.write(json_str + "\n")

    count_sample_theta = 0

    count_prime = 0
    c_missed = 0
    c_done = 0
    oot_file_path = f'{out_path}{out_file}'
    not_found_prime = {}
    mention_count_not_found_prime = 0
    gt_not_in_kb_count = 0
    unique_sample = {}
    with open(oot_file_path, 'w') as f:
        for doc in tqdm(corpus):
            try:
                found_gt_id_in_kb = False
                if has_gt:
                    for gt in doc['ground_truth']:
                        gt_id = gt['id']
                        if gt_id in kb:
                            gt_label_id = kb[gt_id]['id']
                            label_title = gt[gt_title_key]
                            label_def = kb[gt_label_id]['def']
                            found_gt_id_in_kb = True
                            break

                    if not found_gt_id_in_kb:
                        gt_label_id = gt_id
                        label_title = None
                        label_def = 'def'
                        gt_not_in_kb_count+=1
                        if skip_sample_if_ent_not_in_kb:
                            continue
                else:
                    gt_label_id = 'id'
                    label_title = ''
                    label_def = 'def'


                

                d = make_blink_format(world, doc, mtokens, kb, map_dict, defi_key, label_title, label_def, gt_label_id)

                smpl_d = deepcopy(d)
                if 'sample_id' in smpl_d:
                    del smpl_d['sample_id']
                nk = ''
                for k in smpl_d:
                    nk+=f'{smpl_d[k]}'
                if nk not in unique_sample:
                    unique_sample[nk]=0
                    f.write(json.dumps(d) + "\n")
                    c_done+=1
        
            except Exception as e:
                print(e)
                c_missed+=1

    if c_missed>0:
        print(f'{c_done} is done, but {c_missed} are not able to convert! this a big issue!')

    if only_ho is True or ho_prime_others_not is True:
        print(f'For {mention_count_not_found_prime} mention prime GT not found,\nUnique GTs not found {len(not_found_prime)}!')

    

    # print(f'created mention corpus  : {c_done} : {oot_file_path}')
    print(f'{count_prime} prime added to the mention corpus')
    print(f'{gt_not_in_kb_count} gt_not_in_kb ')
    



    for cat in cat_wise_acc:
        try:
            cat_wise_acc[cat]['acc'] =round(cat_wise_acc[cat]['correct']/cat_wise_acc[cat]['count'],2)
        except ZeroDivisionError:
            pass

    with open(f'{out_path}cat_wise_acc.json', 'w') as f:
        json.dump(cat_wise_acc, f, indent=2)

    check_duplicate(oot_file_path)

def save_kb_and_id_map(world, onto_path, kb, title_key, defi_key):
    map_dict = {}
    ent_list = []
    id_map={}
    kb_file = f'{onto_path}kb.jsonl'
    with open(kb_file, 'w') as f:
        for i, id in enumerate(kb):
            inc_id = i
            id_map[inc_id]=id
            map_dict[id]=inc_id
            kb[id]['id'] =inc_id
            d = {'id':inc_id, 'title':kb[id][title_key],'text':kb[id][defi_key]}
            json_str = json.dumps(d)
            f.write(json_str + "\n")
            dn = {'idx':f"{world}?curid={inc_id}", 'entity':kb[id][title_key], 'title':kb[id][title_key],'text':kb[id][defi_key]}
            ent_list.append(dn)
    print(f'created kb : {len(ent_list)} : {kb_file}')

    id_map_file=f'{onto_path}id_map.json'
    with open(id_map_file, 'w') as f:
        json.dump(id_map,f)
    print(f'created id map  : {len(id_map)} : {id_map_file}')

    entity_file = f'{onto_path}entity.jsonl'
    with open(entity_file, 'w') as f:
        for ent in ent_list:
            json_str = json.dumps(ent)
            f.write(json_str + "\n")

    return map_dict

def get_word_overlap_count(term_1, term_2):
    words_term_1  = set(term_1.split())
    words_term_2 = set(term_2.split())
    common_words = words_term_1.intersection(words_term_2)
    return len(common_words)

def get_term_similarity(model, term_1, term_2, word_overlap=False):
    if word_overlap:
        return get_word_overlap_count(term_1, term_2)
    else:
        
        # sim_score = round(model.similarity(model.encode(term_1), model.encode(term_2)).item(), 2)

        
        emb1 = model.encode(term_1, convert_to_tensor=True)
        emb2 = model.encode(term_2, convert_to_tensor=True)
        sim_score = round(util.cos_sim(emb1, emb2).item(), 2)
        return sim_score
    
def does_ent_sim_smaller_than_its_parent_child_or_threshhold(model, item_id, relations, kb, kb_prime_def_m, ent_prime):
    ent_old_name = kb_prime_def_m[item_id]['old_name']
    # ent_prime = kb_prime_def_m[item_id]['newly_generated_name']
    ent_score = get_term_similarity(model, ent_old_name, ent_prime)

    score_dict = {}
    scores = []
    for relation in relations:
        if item_id in relation:
            ent_relation = relation[item_id]
            if 'parents' in ent_relation:
                ent_parents = ent_relation['parents']
                for parent_ent_id in ent_parents:
                    parent_ent = kb[parent_ent_id]
                    parent_ent_name = parent_ent['name']
                    parent_score = get_term_similarity(model, parent_ent_name, ent_prime)
                    scores.append(parent_score)
                    score_dict[parent_ent_name]=parent_score
            if 'children' in ent_relation:
                ent_children = ent_relation['children']
                for child_ent_id in ent_children:
                    child_ent = kb[child_ent_id]
                    child_ent_name = child_ent['name']
                    child_score = get_term_similarity(model, child_ent_name, ent_prime)
                    scores.append(child_score)
                    score_dict[child_ent_name]=child_score

    ent_sim_smaller_than_its_parent_child = False
    for scr in scores:
        if ent_score<scr:
            ent_sim_smaller_than_its_parent_child = True
            break

    if ent_score < 0.9:
        ent_sim_smaller_than_its_parent_child = True


    parent_child_score = str({'ent':ent_old_name, 'score':ent_score,'prime':ent_prime, 'parent_child':score_dict}).replace("\\", '')
    
    return ent_sim_smaller_than_its_parent_child, parent_child_score


def get_mesh_relations():
    rfile = ['relations_desc2025.json',
        'relations_pa2025.json',
        'relations_qual2025.json',
        'relations_supp2025.json']
    relations=[]
    for i in rfile:
        with open('data/bc5cdr/onto/'+i) as f:
            data = json.load(f)
        relations.append(data)
    return relations

def get_medic_relations(data_dir, kb_files):
    relations=[]
    for i in kb_files:
        with open(data_dir+i) as f:
            data = json.load(f)
            relation_dict = {}
            for ent in data:
                if isinstance( data[ent]['ParentIDs'], list):
                    ent_rel = [ en.replace('MESH:', '') for en in data[ent]['ParentIDs'] ]
                else:
                    if data[ent]['ParentIDs'] == "MESH:C":
                        continue

                    ent_rel = [ data[ent]['ParentIDs'].replace('MESH:', '') ]

                relation_dict[ent] = {'parents' : ent_rel}
            
            relations.append(relation_dict)

    return relations

def get_animal_science_relations(data_dir, onto):
    kb_file = [f'{onto}_kb.json']
    relations=[]
    for i in kb_file:
        with open(data_dir +i) as f:
            data = json.load(f)
            relation_dict = {}
            for ent in data:
                if isinstance( data[ent]['ParentIDs'], list):
                    ent_rel = [ en.replace('MESH:', '') for en in data[ent]['ParentIDs'] ]
                else:
                    if data[ent]['ParentIDs'] == "MESH:C":
                        continue

                    ent_rel = [ data[ent]['ParentIDs'].replace('MESH:', '') ]

                relation_dict[ent] = {'parents' : ent_rel}
            
            relations.append(relation_dict)

    return relations


def get_relations_from_ontology(onto, kb_filepath):
    rfile = [kb_filepath]
    relations=[]
    for i in rfile:
        with open(i) as f:
            data = json.load(f)
            relation_dict = {}
            for ent in data:
                if isinstance( data[ent]['ParentIDs'], list):
                    ent_rel = [ en.replace('MESH:', '') for en in data[ent]['ParentIDs'] ]
                else:
                    if data[ent]['ParentIDs'] == "MESH:C":
                        continue

                    ent_rel = [ data[ent]['ParentIDs'].replace('MESH:', '') ]

                relation_dict[ent] = {'parents' : ent_rel}
            
            relations.append(relation_dict)

    return relations


def convert_kb(id_pattern, in_filepath, out_filepath):
    with open(in_filepath) as f:
        existing = json.load(f)
    converted = {}
    for ent in existing:
        synonyms = []
        for synm in ent['synonym']:
            synm_cleaned = re.findall(r'"(.*?)"', synm)[0]
            synonyms.append(synm_cleaned) 
        ParentIDs = []
        for parent in ent['parent_of']:
            pid_ent = re.findall(id_pattern, parent)
            ParentIDs.extend(pid_ent) 
        if len(ParentIDs) != len( ent['parent_of']):
            raise ValueError("len(ParentIDs) != len( ent['parent_of'])")
        is_a_rel = re.findall(id_pattern, ent['is_a'])
        d =  {
            "id": ent['id'],
            "name": ent['name'],
            "def": ent['def'],
            "synonyms": synonyms,
            "altdiseaseid": [],
            "ParentIDs": ParentIDs,
            "is_a": is_a_rel
            }
        converted[ent['id']] = d
    with open(out_filepath, 'w') as f:
       json.dump(converted, f, indent=1)

