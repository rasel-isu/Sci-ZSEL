from copy import deepcopy
import errno
import json
import os
import pickle
import random
import shutil
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils import CONFIG

def show_len(filepah):
    with open(filepah, 'r') as f:
        corpus = json.load(f)
    print(f'{len(corpus)} : {filepah}')


def blink_retriver_to_res(
        onto, split_name, out_dataset, blink_retriver_file, kb_file_path, 
        out_file, blink_raw_data_path, made_from_promtel=True, is_prime=True, 
        ho_prime_others_not=False,
        m3=False,
        m4=False,
        m5=False,
        m6=False):

    with open(blink_retriver_file, 'r') as f:
        blink_retriever = json.load(f)
    
    with open(kb_file_path, 'r') as f:
        kb = json.load(f)

    with open(f'{blink_raw_data_path}id_map.json', 'r') as f:
        id_map = json.load(f)

    if is_prime:
        if m3:
            prime_kb_file = f"{onto}_prime_{split_name}_newly_generated.json"
        elif m4:
            prime_kb_file = f"top_1_from_biencoder_newly_generated.json"
        elif m5:
            prime_kb_file = f"e3_from_parent_child_of_e1_newly_generated.json"
        elif m6:
            prime_kb_file = f"mention_overlap_unq_terms_newly_generated.json"

        with open(blink_raw_data_path+prime_kb_file, 'r') as f:
            prime_data = json.load(f)
            prime_kb = {}
            for p in prime_data:
                prime_kb[p['document_id']] = {
                    "old_name":p['old_name'], "title": p['newly_generated_name'], "text": p['def'], "document_id": p['document_id']
                }

        out_prime_kb = f"kb/{prime_kb_file}"
        with open(f'{out_prime_kb.replace(".json", "_prime.json")}', 'w') as f:
            json.dump(prime_kb, f, indent=1)
    
    

    uncon_prime = 0
    convert_corpus = []
    convert_corpus_ho = []
    cln = {}
    for doc in tqdm(blink_retriever):
        gt_id = id_map[str(doc['mention_data']['label_id'])]
        if gt_id not in kb:
            continue

        if not ho_prime_others_not:
            if is_prime:
                if gt_id not in prime_kb:
                    continue
        

        # candidates = list(doc["unique_triple"].keys())
        candidates = []
        for c in doc['retriever_predictions']:
            candidates.append(id_map[str(c['id'])])


        if gt_id not in candidates:
            candidates = candidates[:len(candidates)-1] + [gt_id]
        cln[len(candidates)]=0
        mdata = {
            'kb_id': gt_id,
            'candidates': candidates
            }
        
        m = doc['mention_data']
        mention = m['mention']
        context_left = m['context_left'].strip()
        context_right = m['context_right'].strip()
        if context_left == '' and context_right != '':
            mention_context = '[E1] '+mention.strip()+' [\\E1] '+context_right
        elif context_left != '' and context_right == '':
            mention_context = context_left+' [E1] '+mention.strip()+' [\\E1]'
        elif context_left == '' and context_right == '':
            mention_context = '[E1] '+mention.strip()+' [\\E1]'
        else:
            mention_context = m['context_left'].strip() + ' [E1] '+mention.strip()+' [\\E1] '+m['context_right'].strip()
        mention_context = mention_context.strip()
        
        d = {
            'sample_id':doc['mention_data']['sample_id'],
            'text':mention_context,
            'mention_data':mdata
        }


        convert_corpus.append(d)

        ml = mention.lower().strip()
        gttl = kb[gt_id]['title']

        if is_prime:
            if ml == gttl:
                if gt_id in prime_kb:
                    convert_corpus_ho.append(d)
                else:
                    uncon_prime+=1

    # print(cln)
    if uncon_prime > 0:
        print(f'! {uncon_prime} : Unable to convert prime entities')
    with open(out_file, 'w') as f:
        json.dump(convert_corpus, f, indent=1)
    show_len(out_file)
    

def copyanything(src, dst):
    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: 
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise
        
def create_data_from_blink(onto):

    split_name = 'train'
    ret_path = f'../'
    out_dir_root = f"../{CONFIG['data_dir']}/res_format/{split_name}/"
    # os.makedirs(f'{out_dir_root}/other_files/', exist_ok=True)

    exps = {
        'original_data':None,
        # "(m1_e1)":'m1',
        # "(m1_e1)U(m3_e1)U(m4_e2)":None,
        # "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime":None,
        "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e":None,
        # "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e":None,
        # "synonym":None,
        }
    
    # if onto =='bc5cdr':
    #     kb_file = f"kb/mesh_kb.json"
    #     copyanything(f'{ret_path}data/{onto}/onto/', f'{out_dir_root}/other_files')
    # elif onto =='ncbi':
    #     kb_file = f"kb/medic_kb.json"
    #     copyanything(f'{ret_path}data/{onto}/onto/', f'{out_dir_root}/other_files')
    # elif onto in ['cmo', 'vt', 'lpt']:

    # copyanything(f'{ret_path}datasets/{onto}/', f'{out_dir_root}/other_files')

     
    with open(f"../{CONFIG['data_dir']}/{CONFIG['kb_file']}", 'r') as f:
        ents = json.load(f)

    res_ent = {}
    for e in ents:
        res_ent[e] = {
            'document_id':e,
            'title':ents[e]['name'],
            'text':ents[e]['def'],
        }
    kb_file = f'{out_dir_root}/{CONFIG["kb_file"]}'
    with open(kb_file, 'w') as f:
        json.dump(res_ent, f, indent=1)
    
    # shutil.copy2(f'{ret_path}datasets/{onto}/test_grag.json', f'{out_dir_root}/other_files/test_grag.json')

    for exp in exps:
        out_dir = f"{out_dir_root}{exp}/"
        os.makedirs(out_dir, exist_ok=True)

        if exp=='original_data':
            out_dataset = 'test'
        else:
            out_dataset = 'train'
        
        outfile = f"{out_dir}{out_dataset}.json"
        blink_raw_data_path = f"../{CONFIG['data_dir']}/blink_format/{split_name}/{exp}/"
        blink_retriver_file = f'../{CONFIG["saved_model_dir"]}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'

        if exps[exp] == 'm1':
            blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, is_prime=False)
        elif exps[exp] == 'm4':
            blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, m4=True)
        else:
            blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, is_prime=False)
            
if __name__ == "__main__":
    create_data_from_blink(CONFIG['world'])

