from copy import deepcopy
import json
import os
import shutil
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from utils import CONFIG, check_duplicate, get_animal_science_relations, get_medic_relations, get_mesh_relations, get_relations_from_ontology, read_jsonl, make_blink_format, get_category,save_kb_and_id_map, \
is_accuratly_pseudo_labelled,gt_mention_category_count, does_ent_sim_smaller_than_its_parent_child_or_threshhold


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


def grag_to_blink_for_synonym_only(
        input_from,kb_file,output_dir,synonym_key, has_gt=True
    ):

    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(f'{input_from}id_map.json', f'{output_dir}id_map.json')
    shutil.copyfile(f'{input_from}entity.jsonl', f'{output_dir}entity.jsonl')
    shutil.copyfile(f'{input_from}kb.jsonl', f'{output_dir}kb.jsonl')

    mentions = []
    with open(f'{input_from}train.jsonl') as f:
        lines = f.readlines()
        for line in lines:
            try:
                mentions.append(json.loads(line))
            except:
                print('\n\n\n')
                print(line)
                print('\n')
    print(len(mentions))

    with open(kb_file) as f:
        entities = json.load(f)
    entities = list(entities.values())

    with open(f'{input_from}id_map.json') as f:
        id_map = json.load(f)
        id_map = {v:k for k,v in id_map.items()}

    synonyms = []
    count = 0 # number of mentions that map to more than 1 entity

    sample_correctness = {}
    for mention in mentions:
        new_mention = mention.copy()
        mention_title = new_mention['mention']
        sample_correctness[new_mention['sample_id']]={'correct':False}
        add = 0
        for entity in entities:
            syns = entity[synonym_key]
            if isinstance(syns, str):
                syns = [syns]
            
            for i in range(len(syns)):
                if syns[i].lower() == mention_title.lower() or mention_title.lower() == entity['name'].lower():
                    new_mention['label_title'] = entity['name']
                    new_mention['label'] = entity['def']
                    new_mention['label_id'] = int(id_map[entity['id']])
                    new_mention['gt'] = mention['label_id']
                    synonyms.append(new_mention)
                    add += 1

                    break

        if add > 1:
            count += 1

    cat_wise_acc = {
        "HO":  {'count': 0, 'acc': 0, 'correct': 0},
        "MINT":{'count': 0, 'acc': 0, 'correct': 0},
        "LO":  {'count': 0, 'acc': 0, 'correct': 0},
        "NO":  {'count': 0, 'acc': 0, 'correct': 0}}


    out_file = f"{output_dir}/train.jsonl"
    unique_sample = {}
    with open(out_file, "w") as f:
        for item in synonyms:
            d = deepcopy(item)
            d.pop('sample_id', None)
            gt = d.pop('gt', None)
            nk = ''
            for k in d:
                nk+=f'{d[k]}'
            if nk not in unique_sample:
                unique_sample[nk]=0
                f.write(json.dumps(item) + "\n")

                category = get_category(item['mention'], item['label_title'])
                cat_wise_acc[category]['count'] += 1
                if has_gt:
                    if gt == item['label_id']:
                        cat_wise_acc[category]['correct'] += 1

    check_duplicate(out_file)

    print(len(synonyms))
    print("Number of mentions that map to more than 1 entity:", count)


    overall = {'count': 0, 'acc': 0, 'correct': 0}
    for cat in cat_wise_acc:
        overall['count']   += cat_wise_acc[cat]['count']
        overall['correct'] += cat_wise_acc[cat]['correct']
        try:
            cat_wise_acc[cat]['acc'] = round(cat_wise_acc[cat]['correct'] / cat_wise_acc[cat]['count'], 2)
        except ZeroDivisionError:
            pass
    try:
        overall['acc'] = round(overall['correct'] / overall['count'], 2)
    except ZeroDivisionError:
        pass
    cat_wise_acc['OVERALL'] = overall

    with open(f'{output_dir}/cat_wise_acc.json', 'w') as f:
        json.dump(cat_wise_acc, f, indent=2)


def grag_to_blink_for_all_m(
        source_dir, 
        source_file, 
        kb_file_path,
        title_key, 
        defi_key, 
        world, 
        split_name,
        out_file,
        onto_path=None,
        mtokens=['[MENTION_START]', '[MENTION_END]'],
        plural=False,
        m3=False,
        m4=False,
        m4_64=False,
        m5=False,
        m6=False,
        remove_samller_e=False,
        label_will_be_gt = False,
        exclude_ho = False,
        only_correct_label = False,
        multi_prime = True
        ):

    with open(source_dir+source_file, 'r') as f:
        corpus = json.load(f)
    print(f'corpus : {len(corpus)} : {source_dir+source_file}')
    with open(kb_file_path, 'r') as f:
        kb = json.load(f)
    print(f'kb : {len(kb)} : {kb_file_path}')

    if not onto_path:
        onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/"
    os.makedirs(onto_path, exist_ok=True)
    
    if m3:
        if '_rm_sm_e' in onto_path:
            prime_path = onto_path.replace("(m3_e1)_multi_prime_rm_sm_e", "(m1_e1)")
        else:
            prime_path = onto_path.replace("(m3_e1)_multi_prime", "(m1_e1)")
        prime_kb_file = f'{prime_path}m1_e1_unq_ents_newly_generated.json'
    elif m4:
        if '_rm_sm_e' in onto_path:
            prime_path = onto_path.replace("(m4_e2)_multi_prime_rm_sm_e", "original_data")
        else:
            prime_path = onto_path.replace("(m4_e2)_multi_prime", "original_data")
        prime_kb_file = f'{prime_path}top_1_from_biencoder_newly_generated.json'
    elif m4_64:
        prime_kb_file = f'{onto_path}top_64_from_biencoder_newly_generated.json'
    elif m5:
        prime_kb_file = f'{onto_path}e3_from_parent_child_of_e1_newly_generated.json'
    elif m6:
        prime_kb_file = f'{onto_path}mention_overlap_unq_terms_newly_generated.json'
    
    kb_prime_def = get_prime_kb(onto_path, world, split_name, prime_kb_file)
    map_dict = save_kb_and_id_map(world, onto_path, deepcopy(kb), title_key, defi_key)

    if remove_samller_e:
        cache_dir = "/lustre/hdd/LAS/qli-lab/rasel/kgllama/models/LLaMA-HF/"
        base_model = 'FremyCompany/BioLORD-2023-M'

        dir_name = base_model.replace('/', '_')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(base_model, cache_folder=cache_dir+'model/'+dir_name).to(device)

        if world=='bc5cdr':
            relations = get_mesh_relations()
        elif world=='ncbi_disease':
            relations = get_medic_relations(CONFIG['data_dir'], [CONFIG['kb_file']])
        elif world in ['cmo', 'vt', 'lpt',]:
            relations = get_animal_science_relations(CONFIG['data_dir'], world)
        elif world in ['cometa']:
            relations = get_relations_from_ontology(world, kb_file_path) 

    cat_wise_acc = {
            "HO":{'count':0, 'acc':0, 'correct':0}, 
            "MINT":{'count':0, 'acc':0, 'correct':0},
            "LO":{'count':0, 'acc':0, 'correct':0},
            "NO":{'count':0, 'acc':0, 'correct':0}}
    

    men_eql_prime_count=0
    oot_file_path = f'{onto_path}{out_file}'
    incorrectly_labelled_samples = []
    men_lbl_gt = []
    saved_samples = []

    unique_sample = {}
    with open(oot_file_path, 'w') as f:
        for doc in tqdm(corpus):
            # gt_id = doc['ground_truth']['id']
            # if gt_id not in kb:
            #     continue

            mention = doc['mention'].lower().strip()
            
            for item_id in kb_prime_def:
                item_new_name = kb_prime_def[item_id]['newly_generated_name'].lower().strip()
                item_old_name = kb_prime_def[item_id]['old_name'].lower().strip()
                if not plural:

                    has_matching = False
                    if multi_prime:
                        primes = [i.lower().strip() for i in item_new_name.split(',')]
                        if mention in primes:
                            has_matching = True 
                            ent_prime = mention

                    else:
                        if mention == item_new_name:
                            has_matching = True
                            ent_prime = mention

                    if has_matching:
                        gt_id_list = [i['id'] for i in doc['ground_truth'] ]
                        if label_will_be_gt:
                            label_id = doc['ground_truth']['id']
                            label_title = doc['ground_truth']['title']
                            label_def = kb[label_id]['def']
                        else:
                            label_title = kb_prime_def[item_id]['old_name']
                            label_def = kb_prime_def[item_id]['def']
                            label_id = kb_prime_def[item_id]['document_id']

                        if exclude_ho:
                            if mention == label_title.lower().strip():
                                break
                        
                        
                        # if doc['ground_truth'][0]['title'] is None:
                        #     continue
                        # ground_truth = doc['ground_truth'][0]['title'].lower().strip()
                        # if mention==ground_truth:
                        #     continue
                        


                        
                        men_eql_prime_count+=1
                        d = make_blink_format(world, doc, mtokens, kb, map_dict, defi_key, 
                                                label_title, label_def, label_id)
                        
                        if remove_samller_e:
                            is_ent_sim_smaller_than_its_parent_child,parent_child_score = does_ent_sim_smaller_than_its_parent_child_or_threshhold(
                                model, item_id, relations, kb, kb_prime_def, ent_prime)
                            if is_ent_sim_smaller_than_its_parent_child:
                                break

                        
                        if d not in saved_samples:
                            
                            # Analysis
                            category = get_category(mention, label_title)
                            cat_wise_acc[category]['count']+=1

                            gt_id_list = [i['id'] for i in doc['ground_truth']]

                            if item_id not in kb:
                                break

                            pseudo_ent = kb[item_id]
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
                                    break
                            # Analysis


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

                                break

                        # else:
                        #     break
                else:
                    subcat = check_plural(mention, item_new_name)
                    if subcat == "plural":

                        if label_will_be_gt:
                            label_id = doc['ground_truth']['id']
                            label_title = doc['ground_truth']['title']
                            label_def = kb[label_id]['def']
                        else:
                            label_title = kb_prime_def[item_id]['old_name']
                            label_def = kb_prime_def[item_id]['def']
                            label_id =kb_prime_def[item_id]['document_id']
                            
                        # if mention!=ground_truth:
                        if exclude_ho:
                            if mention == label_title.lower().strip():
                                continue

                        men_eql_prime_count+=1
                        # if label_title.lower().strip() != ground_truth:
                        #     raise ValueError("label_title and ground_truth should be same!") # this is only for checking

                        d = make_blink_format(world, doc, mtokens, kb, map_dict, defi_key, 
                                                label_title, label_def, label_id)
                        
                        if remove_samller_e:
                            is_ent_sim_smaller_than_its_parent_child,parent_child_score = does_ent_sim_smaller_than_its_parent_child_or_threshhold(model, item_id,relations, kb, kb_prime_def)
                            if is_ent_sim_smaller_than_its_parent_child:
                                break

                        if d not in saved_samples:
                            saved_samples.append(d) 
                            f.write(json.dumps(d) + "\n")
                            category = get_category(mention, label_title)
                            cat_wise_acc[category]['count']+=1
                            if item_old_name == ground_truth:
                                cat_wise_acc[category]['correct']+=1
                            break
                        # else:
                        #     break


    print(f'{men_eql_prime_count} : samples')
    check_duplicate(oot_file_path)
    
    overall = {'count':0, 'acc':0, 'correct':0}
    for cat in cat_wise_acc:
        try:
            overall['count']+=cat_wise_acc[cat]['count']
            overall['correct']+=cat_wise_acc[cat]['correct']
            cat_wise_acc[cat]['acc'] =round(cat_wise_acc[cat]['correct']/cat_wise_acc[cat]['count'],2)
        except ZeroDivisionError:
            pass
    try:
        overall['acc'] = round(overall['correct']/overall['count'],2)
    except ZeroDivisionError:
        pass
    
    
    cat_wise_acc['OVERALL'] = overall
    with open(f'{onto_path}cat_wise_acc.json', 'w') as f:
        json.dump(cat_wise_acc, f, indent=2)
    

    counts = gt_mention_category_count(f"{onto_path}", out_file)


    if only_correct_label:
        with open(f'{onto_path}incorrectly_labelled_samples.jsonl', 'w') as f:
            f.write('\n'.join(incorrectly_labelled_samples))

        with open(f'{onto_path}mention_label_gt.json', 'w') as f:
            json.dump(men_lbl_gt, f, indent=1)

def merge_dataset(settings, splitname, out_path, out_file):
    all_data = []

    for setting in settings:
        data_dir = f"{CONFIG['data_dir']}/blink_format/{splitname}/{setting}/"
        data = read_jsonl(f'{data_dir}train.jsonl')
        all_data+=data

    os.makedirs(out_path, exist_ok=True)
    oot_file_path = f'{out_path}{out_file}'
    unique_sample = []
    non_overlap_data = []

    with open(oot_file_path, 'w') as f:
        for d in all_data:
            if d not in unique_sample:
                unique_sample.append(d)
                f.write(json.dumps(d) + "\n")
            d_cpy = deepcopy(d)

            del d_cpy['sample_id']
            if d_cpy not in non_overlap_data:
                non_overlap_data.append(d_cpy)

    saved_data = read_jsonl(oot_file_path)
    print(f'{len(saved_data)} : {oot_file_path}')
    print(f'{len(unique_sample)} : unique samples')
    print(f'{len(non_overlap_data)} : d_cpy samples')
    check_duplicate(oot_file_path)

    input_from = f"{CONFIG['data_dir']}/blink_format/{splitname}/{settings[0]}/"
    shutil.copyfile(f'{input_from}id_map.json', f'{out_path}id_map.json')
    shutil.copyfile(f'{input_from}entity.jsonl', f'{out_path}entity.jsonl')
    shutil.copyfile(f'{input_from}kb.jsonl', f'{out_path}kb.jsonl')
  

def pair_construction(kb_file_path, world, split_name, source_file):
    print(f"{'_'*20}{world}{'_'*20}")
    
    # Construction from LLM-generated alias
    #E_EM
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/(m3_e1)_multi_prime/"
    grag_to_blink_for_all_m(source_dir=CONFIG['data_dir'],source_file=source_file, 
            kb_file_path=kb_file_path,title_key='name',defi_key='def',world=world, 
            split_name=split_name,out_file='train.jsonl',onto_path=onto_path, m3=True, multi_prime=True)
    
    # E_EM after Ontology-aware Filtering
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/(m3_e1)_multi_prime_rm_sm_e/"
    grag_to_blink_for_all_m(source_dir=CONFIG['data_dir'],source_file=source_file, 
            kb_file_path=kb_file_path,title_key='name',defi_key='def',world=world, 
            split_name=split_name,out_file='train.jsonl',onto_path=onto_path,remove_samller_e=True, m3=True, multi_prime=True)

    # E_BT after
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/(m4_e2)_multi_prime/"
    grag_to_blink_for_all_m(source_dir=CONFIG['data_dir'],source_file=source_file, 
            kb_file_path=kb_file_path,title_key='name',defi_key='def',world=world, 
            split_name=split_name,out_file='train.jsonl',onto_path=onto_path, m4=True, multi_prime=True)

    # E_BT after Ontology-aware Filtering
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/(m4_e2)_multi_prime_rm_sm_e/"
    grag_to_blink_for_all_m(source_dir=CONFIG['data_dir'],source_file=source_file, 
            kb_file_path=kb_file_path,title_key='name',defi_key='def',world=world, 
            split_name=split_name,out_file='train.jsonl',onto_path=onto_path,remove_samller_e=True, m4=True, multi_prime=True)


    # Sci-ZSEL w/o filter 
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime/"
    merge_dataset(['(m1_e1)','(m3_e1)_multi_prime',"(m4_e2)_multi_prime"],
                split_name, onto_path, 'train.jsonl')

    # Sci-ZSEL
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e/"
    merge_dataset(['(m1_e1)','(m3_e1)_multi_prime_rm_sm_e',
                   '(m4_e2)_multi_prime_rm_sm_e'], 
                   split_name, onto_path, 'train.jsonl')

   
    # Construction from ontology synonym
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/synonym/"
    grag_to_blink_for_synonym_only(
       f"{CONFIG['data_dir']}/blink_format/{split_name}/original_data/", kb_file_path, onto_path, CONFIG['synonym_key_on_ontology'],  
       CONFIG['has_ground_truth'])
    
    # Sci-ZSEL + Synonym
    onto_path = f"{CONFIG['data_dir']}/blink_format/{split_name}/synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e/"
    merge_dataset(['synonym',
                   '(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e'],
                  split_name, onto_path, 'train.jsonl')


def main():
    world = CONFIG['world']
    kb_file = CONFIG['kb_file']
    data_dir = CONFIG['data_dir']
    kb_file_path = f'{data_dir}/{kb_file}'

    pair_construction(kb_file_path, world, 'train', f'train_grag.json')

if __name__ == "__main__":
    main()

    
