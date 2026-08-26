from copy import deepcopy
import errno
import json
import os
import pickle
import random
import shutil
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ThreadPoolExecutor, as_completed
# from rank_bm25 import BM25Okapi
# import nltk
# from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from data_preparation.utils import summary_for_test_ent_appears_in_train
# nltk.download('punkt_tab', download_dir='venv/nltk_data')

def convert_mesh_onto():

    dir_path = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/graphrag-with-neo4j/DATASET/mesh/'
    meshfile_path = f'{dir_path}ontology_desc.json'
    with open(meshfile_path, 'r') as f:
        ontology_desc = json.load(f)
    meshfile_path =  f'{dir_path}ontology_supp2025.json'
    with open(meshfile_path, 'r') as f:
        ontology_supp2025 = json.load(f)
    meshfile_path =  f'{dir_path}ontology_qual2025.json'
    with open(meshfile_path, 'r') as f:
        ontology_qual2025 = json.load(f)
        
    data = ontology_desc+ontology_supp2025+ontology_qual2025
    
    onto = {}
    for o in data:
        onto[o['id']] = {'title': o['name'], 'text': o['def'], 'document_id': o['id']}
    
    with open(f'data/mesh_kb.json', 'w') as f:
        json.dump(onto, f, indent=2)

def convert_medic_onto():
    dir_path = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/data/ncbi/onto/'
    medic_file_path = f'{dir_path}only_medic_def.json'
    with open(medic_file_path, 'r') as f:
        data = json.load(f)
    onto = {}
    for o in data:
        item = data[o]
        onto[o] = {'title': item['name'], 'text': item['def'], 'document_id': item['id']}
    
    with open(f'kb/medic_kb.json', 'w') as f:
        json.dump(onto, f, indent=2)

def convert_corpus(file, out_file,
                    dir = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/',
                    kb_file_path = 'kb/mesh_kb.json'
                   ):
    with open(dir+file, 'r') as f:
        corpus = json.load(f)
    with open(kb_file_path, 'r') as f:
        kb = json.load(f)
    mesh_id_list = list(kb.keys())
    
    convert_corpus = []
    for doc in tqdm(corpus):

        if doc['ground_truth']['id'] not in mesh_id_list:
            continue
        
        candidates = random.sample(mesh_id_list, 63)
        candidates.append(doc['ground_truth']['id'])
        mdata = {
            'kb_id': doc['ground_truth']['id'],
            'candidates': candidates
            }

        convert_corpus.append({
            'text':doc['mention_context'].replace('[MENTION_START]', '[E1]').replace('[MENTION_END]', '[\\E1]'),
            'mention_data':mdata
        })
        with open(out_file, 'w') as f:
            json.dump(convert_corpus, f, indent=1)



# kb_file_path = 'kb/umls_kb.json'
# with open(kb_file_path, 'r') as f:
#     mesh_kb = json.load(f)
# mesh_id_list = list(mesh_kb.keys())

def process_document(doc):
    if doc['ground_truth']['id'] not in mesh_id_list:
        return None
    
    candidates = random.sample(mesh_id_list, 63)
    candidates.append(doc['ground_truth']['id'])
    mdata = {
        'kb_id': doc['ground_truth']['id'],
        'candidates': candidates
    }
    
    processed_doc = {
        'text': doc['mention_context'].replace('[MENTION_START]', '[E1]').replace('[MENTION_END]', '[\\E1]'),
        'mention_data': mdata
    }
    return processed_doc

def convert_corpus_parallely(file, out_file,
                    dir='/lustre/hdd/LAS/qli-lab/rasel/graphrag/graphrag-with-neo4j/DATASET/Prompt-BioEL/dataset/',
                    kb_file_path='kb/mesh_kb.json'):
    
    with open(dir + file, 'r') as f:
        corpus = json.load(f)
        
    convert_corpus = process_map(process_document, corpus, max_workers=128)

    with open(out_file, 'w') as f:
        json.dump(convert_corpus, f, indent=2)

def convert_prime_data_to_train_format(gen_title_file, new_kb_file,
                    old_kb_file_path):
    
    with open(gen_title_file, 'r') as f:
        corpus = json.load(f)

    with open(old_kb_file_path, 'r') as f:
        kb = json.load(f)
        id_list = list(kb.keys())

    unable_gen = ["can't", "couldn't"]
    c_prime=0
    c_failed=0
    ents = {}
    ents_failed = {}
    convert_kb = {}
    for doc in tqdm(corpus):
        gtid = doc['ground_truth']['id']
        ents[gtid] = None
        if gtid in id_list:
            c_prime+=1
            no_cntx = doc['ground_truth']['no_cntx']
            unable_flag = False
            for w in unable_gen:
                if w in no_cntx.lower():
                    unable_flag = True
                    break
            if unable_flag:
                c_failed+=1
                ents_failed[gtid]=None
                continue

            kb[gtid]['title'] = no_cntx
            convert_kb[gtid] = kb[gtid]

    print(f'Total mention sample : {c_prime}\nFailed : {c_failed}')
    print(f'Unique entites : {len(ents)}\nFailed : {len(ents_failed)}')

    with open(new_kb_file, 'w') as f:
        json.dump(convert_kb, f, indent=2)

def show_len(filepah):
    with open(filepah, 'r') as f:
        corpus = json.load(f)
    print(f'{len(corpus)} : {filepah}')


def convert_all(onto, splitname, corpus_dir, kbpath):
    dir_data = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/'
    in_filename = f'{corpus_dir}/{splitname}_grag_generated_ho.json'
    otfile = f'data/{onto}_{splitname}_ho.json'
    otprime_kb = f'kb/{onto}_prime_{splitname}.json'
    ho_file = dir_data+ in_filename
    convert_corpus(in_filename, dir=dir_data, out_file=otfile)
    show_len(ho_file)
    show_len(otfile)
    convert_prime_data_to_train_format(ho_file, otprime_kb, kbpath)
    show_len(otprime_kb)

    dev_filename = f'{corpus_dir}/dev_grag.json'
    dev_otfile = f'data/{onto}_dev.json'
    convert_corpus(dev_filename, dir=dir_data, out_file=dev_otfile)
    show_len(dev_otfile)

def get_candidates_from_retriever(doc, id_map):
    top_k = list(doc["unique_triple"].keys())
    ids = []
    for id in top_k:
        ids.append(id_map[id])
    return ids

def remove_sample_if_gt_not_in_prime(filepath, prime_kbpath):
    with open(filepath) as f:
        mentions = json.load(f)
    with open(prime_kbpath) as f:
        kb = json.load(f)
    removed = []
    for m in mentions:
        gtid = m['mention_data']['kb_id']
        if gtid in kb:
            removed.append(m)

    with open(filepath.replace('.json', '_removed_if_not_in_prime.json'), 'w') as f:
        json.dump(removed, f, indent=1)

def check_correctness_of_prepared_data(res_file, prompt_el_file):
    with open(prompt_el_file, encoding="utf-8") as f:
        pr_bio_el_data = [json.loads(line) for line in f]
    with open(f'{res_file}') as f:
        res_data = json.load(f)
    len_res = len(res_data)
    print(f'{len(pr_bio_el_data)} : {prompt_el_file}')
    print(f'{len_res} : {res_file}')
    if len(pr_bio_el_data) != len_res:
        print(f'Sample size is not same! (prompt_el : {len(pr_bio_el_data)}, res_data : {len_res})')
        matched_items = []
        matched = 0
        for item_res in res_data:
            men_res = item_res['text'].split('[E1]')[1].split('[\\E1]')[0].strip()
            men_cntx_res = item_res['text'].replace('[\\E1]', '[/E1]')
            gt_res = item_res['mention_data']['kb_id']
            not_found = True

            for item_prm in pr_bio_el_data:
                men_cntx_prm = item_prm['text']
                
                if men_cntx_prm==men_cntx_res:
                    men_prm = item_prm['mention_data'][0]['mention']
                    if men_prm==men_res:
                        gt_prm = item_prm['mention_data'][0]['kb_id']
                        if gt_prm==gt_res:
                            matched+=1
                            not_found = False
                            matched_items.append(item_res)
                            break
                        else:
                            ids = gt_prm.split('|')
                            if gt_res in ids:
                                matched+=1
                                not_found = False
                                matched_items.append(item_res)
                                break

            if not_found:
                for item_prm in pr_bio_el_data:
                    men_prm = item_prm['mention_data'][0]['mention']
                    if men_res==men_prm:
                        print(0)
            
                
        if len_res==matched:
            print('All the sample of ReS were from Prompt-Bio-EL')
            
        else:

            # with open('res_data.txt', 'w') as f:
            #     for i in res_data:
            #         f.write(f'{i}\n')
            # with open('matched_items.txt', 'w') as f:
            #     for i in matched_items:
            #         f.write(f'{i}\n')

            print(f'Not all the samples are matched -> Res has {len_res} but sample matched : {matched}')
           

    

def blink_retriver_to_res(
        ret_path, 
        onto, split_name, out_dataset, blink_retriver_file, kb_file_path, 
        out_file, blink_raw_data_path, made_from_promtel=True, is_prime=True, 
        ho_prime_others_not=False,
        m3=False,
        m4=False,
        m5=False,
        m6=False):
    
    
    # ret_path = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/'
    # ret_path = f'/lustre/hdd/LAS/qli-lab/fssamia/graphrag/blink_rs/'

    with open(ret_path+blink_retriver_file, 'r') as f:
        blink_retriever = json.load(f)
    
    with open(kb_file_path, 'r') as f:
        kb = json.load(f)

    with open(f'{ret_path}{blink_raw_data_path}id_map.json', 'r') as f:
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

        with open(ret_path+blink_raw_data_path+prime_kb_file, 'r') as f:
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

    print(cln)
    print(f'! {uncon_prime} : Unable to convert prime entities')

    prompt_el_file = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{out_dataset}.json'

    with open(out_file, 'w') as f:
        json.dump(convert_corpus, f, indent=1)
    show_len(out_file)
    
    
    # if made_from_promtel:
    #     check_correctness_of_prepared_data(out_file, prompt_el_file)

    # if split_name=='test':
    #     filename_ind = -10
    # elif split_name=='train':
    #     filename_ind = -11
    # if with_ho and is_prime is True:
    #     if out_file[filename_ind:] == f'_{split_name}.json':
    #         out_file_ho = out_file.replace(f'_{split_name}.json', f'_{split_name}_ho.json')
    #         with open(out_file_ho, 'w') as f:
    #             json.dump(convert_corpus_ho, f, indent=1)
    #         show_len(out_file_ho)
    #         if made_from_promtel:
    #             check_correctness_of_prepared_data(out_file_ho, prompt_el_file)
    



def merge_train_test(onto, ho=True):
    test_kb_file = f'kb/{onto}_prime_test.json'
    train_kb_file = f'kb/{onto}_prime_train.json'
    merged_kb_file = f'kb/{onto}_prime_train_and_test.json'
    with open(test_kb_file, 'r') as f:
        test_kb_prime = json.load(f)
    with open(train_kb_file, 'r') as f:
        train_kb_prime = json.load(f)
    c = 0
    for i in test_kb_prime:
        if i in train_kb_prime:
            c+=1
    print(f'{c} prime entities from test set is also appears in train set!')
    merged_kb = {**test_kb_prime, **train_kb_prime}
    with open(merged_kb_file, 'w') as f:
        json.dump(merged_kb, f, indent=1)

    print(f'After merging train test prime KB')
    show_len(test_kb_file)
    show_len(train_kb_file)
    show_len(merged_kb_file)

    if ho:
        test_file = f"data/{onto}_test_ho.json"
        train_file = f"data/{onto}_train_ho.json"
        outfile = f'data/{onto}_train_and_test_ho.json'
    else:
        test_file = f"data/{onto}_test.json"
        train_file = f"data/{onto}_train.json"
        outfile = f'data/{onto}_train_and_test.json'

    with open(test_file, 'r') as f:
        test = json.load(f)
    with open(train_file, 'r') as f:
        train = json.load(f)
    with open(outfile, 'w') as f:
        json.dump(test+train, f, indent=1)

    print(f'After merging train test mention corpus')
    show_len(test_file)
    show_len(train_file)
    show_len(outfile)

def get_bm25(kb_file):
    with open(kb_file, 'r') as f:
        kb = json.load(f)

    tokenized_corpus = []
    doc_ids = []
    for doc_id, content in tqdm(kb.items()):
        merged_text = content["title"] + " " + content["text"]
        tokens = word_tokenize(merged_text.lower())
        tokenized_corpus.append(tokens)
        doc_ids.append(doc_id)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_ids

def get_candidates_bm25(query, bm25, doc_ids):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    results = list(zip(doc_ids, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    ids = []
    for doc_id, score in results[:64]:
        ids.append(doc_id)
    return ids

def get_hf_tf_encoding(kb_file):
    base_model = 'emilyalsentzer/Bio_ClinicalBERT'
    dir_name = base_model.replace('/', '_')
    cache_dir = "/lustre/hdd/LAS/qli-lab/rasel/kgllama/models/LLaMA-HF/"
    model = SentenceTransformer(base_model, cache_folder=cache_dir+'model/'+dir_name)
    with open(kb_file, 'r') as f:
        kb = json.load(f)
    encode_file_path = kb_file.replace('.json', '.pkl')
    if os.path.exists(encode_file_path):
        print("Loading embeddings from file...")
        with open(encode_file_path, 'rb') as f:
            doc_enc = pickle.load(f)
    else:
        print("Encoding documents and saving to file...")
        doc_enc = {}
        for doc_id, content in tqdm(kb.items()):
            merged_text = content["title"] + " " + content["text"]
            enc = model.encode(merged_text)
            doc_enc[doc_id] = enc
        with open(encode_file_path, 'wb') as f:
            pickle.dump(doc_enc, f)
    return model, doc_enc

def get_candidates_tf(mention, model, encodings):
    mention_enc = model.encode(mention)
    scores = []
    for e in encodings:
        sim = round( model.similarity(mention_enc, encodings[e]).item(), 2)
        scores.append({'id':e, 'score':sim})

    results = sorted(scores, key=lambda x: x['score'], reverse=True)

    with open('sample.json', 'w') as f:
        json.dump({'mention':mention, 'scores':results}, f, indent=1)
    input("scores : ")

    ids = []
    k = 64 
    for i, ent in enumerate(results):
        ids.append(ent['id'])
        if i+1==k:
            break
    return ids

def create_data_using_pairwise_retriever(kb_file, mc_file, outfile, bm25=True):
    with open(mc_file, 'r') as f:
        mc = json.load(f)
    if bm25:
        bm25, doc_ids = get_bm25(kb_file)
    else:
        model, encodings = get_hf_tf_encoding(kb_file)
    new_m = []
    for i in tqdm(mc):
        mention_context = i['text']
        mention = mention_context.split("[E1]")[1].split("[\E1]")[0].strip()
        if bm25:
            cand = get_candidates_bm25(mention, bm25, doc_ids)
        else:
            cand = get_candidates_tf(mention, model, encodings)
        kb_id = i['mention_data']['kb_id']
        if kb_id not in cand:
            cand = cand[:64 - 1] + [kb_id]
        i['mention_data']['candidates'] = cand
        new_m.append(i)

    with open(outfile, 'w') as f:
        json.dump(new_m, f, indent=1)

def prepare_for_prime_def(
    old_prime_file, 
    new_prime_file, 
    train_data_file
    ):
    with open(old_prime_file, 'r') as f:
        old_prime = json.load(f)
    with open(new_prime_file, 'r') as f:
        new_prime_data = json.load(f)
        new_prime = {}
        for i in new_prime_data:
            new_prime[i['document_id']] = i

    
    replaced = {}
    for o in old_prime:
        if not old_prime[o]['text'] == '':
            old_prime[o]['title'] = new_prime[o]['newly_generated_name']
            replaced[o]=old_prime[o]

    print(
        f'For some reason \nold prime has {len(old_prime)} entity\nnew prime has {len(replaced)} entity'
        )

    with open(old_prime_file.replace('.json', '_defi.json'), 'w') as f:
        json.dump(replaced, f, indent=1)

    with open(train_data_file, 'r') as f:
        train_data = json.load(f)

    new_train_data = []
    for d in train_data:
        kb_id = d['mention_data']['kb_id']
        if kb_id not in replaced:
            continue
        else:
            new_train_data.append(d)

    with open(train_data_file.replace('.json', '_defi.json'), 'w') as f:
        json.dump(new_train_data, f, indent=1)

    print(
        f'For some reason \nold train data has {len(train_data)} mentions\nnew train data has {len(new_train_data)} mentions'
        )
    

def res_shuffle_candidates(res_file):
    with open(res_file, 'r') as f:
        mc = json.load(f)

    convert_corpus = []
    for doc in tqdm(mc):
        candidates = doc["mention_data"]['candidates']
        shuffled_candidates = deepcopy(candidates)
        random.shuffle(shuffled_candidates)
        doc["mention_data"]['candidates'] = shuffled_candidates
        convert_corpus.append(doc)
    with open(res_file.replace('.json', '_shuffled_candidates.json'), 'w') as f:
        json.dump(convert_corpus, f, indent=1)

def copyanything(src, dst):
    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: 
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise
        
def create_data_from_blink(onto,split_name):



    ret_path = f'/lustre/hdd/LAS/qli-lab/rasel/projects/Sci-ZSEL/BLINK/'
    out_dir_root = f"data/blink/zshel_64_context_length/{onto}/{split_name}/"
    os.makedirs(f'{out_dir_root}/other_files/', exist_ok=True)
    

    if onto =='bc5cdr':
        kb_file = f"kb/mesh_kb.json"
        copyanything(f'{ret_path}data/{onto}/onto/', f'{out_dir_root}/other_files')
    elif onto =='ncbi':
        kb_file = f"kb/medic_kb.json"
        copyanything(f'{ret_path}data/{onto}/onto/', f'{out_dir_root}/other_files')
    elif onto in ['cmo', 'vt', 'lpt']:
        copyanything(f'{ret_path}data/{onto}/onto/', f'{out_dir_root}/other_files')

        with open(f'{ret_path}data/{onto}/onto/{onto}_kb.json', 'r') as f:
            ents = json.load(f)
        res_ent = {}
        for e in ents:
            res_ent[e] = {
                'document_id':e,
                'title':ents[e]['name'],
                'text':ents[e]['def'],
            }
        kb_file = f"kb/{onto}_kb.json"
        with open(kb_file, 'w') as f:
            json.dump(res_ent, f, indent=1)
    
    shutil.copy2(f'{ret_path}data/{onto}/test_grag.json', f'{out_dir_root}/other_files/test_grag.json')

    exps = {
        # "(m1_e1)":'m1',
        # '(m4_e2)':'m4',
        # "(m1_e1)U(m3_e1)U(m4_e2)":None,
        # "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime":None,
        # "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e":None,
        # "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e":None,
        # "synonym":None,
        "synonymU(m1_e1)":None,
        'synonymU(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e':None,
        }
    
    for exp in exps:
        out_dir = f"{out_dir_root}{exp}/"
        os.makedirs(out_dir, exist_ok=True)

        if exp=='original_def_prmbl':
            out_dataset = 'test'
        else:
            out_dataset = 'train'
        
        outfile = f"{out_dir}{out_dataset}.json"
        blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
        blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'

        if exps[exp] == 'm1' or exps[exp] == None:
            blink_retriver_to_res(ret_path, onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, is_prime=False)
        elif exps[exp] == 'm3':
            blink_retriver_to_res(ret_path, onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, m3=True)
        elif exps[exp] == 'm4':
            blink_retriver_to_res(ret_path, onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, m4=True)
        elif exps[exp] == 'm5':
            blink_retriver_to_res(ret_path, onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, m5=True)
        elif exps[exp] == 'm6':
            blink_retriver_to_res(ret_path, onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, m6=True)
            



# convert_medic_onto()
# convert_mesh_onto()
# convert_corpus('bc5cdr/test_grag.json', out_file='data/bc5cdr_test.json')
# convert_corpus('ncbi-disease/test_grag.json', out_file='data/ncbi_test.json')
# dir = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/MedMentions/full/data/'
# convert_corpus_parallely('corpus_pubtator_test.json', out_file='data/medmentions_test.json', 
#                dir=dir, kb_file_path=kb_file_path)

# convert_corpus('test_grag.json', 'data/cometa_test.json',
#                     dir = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/cometa/Prompt-BioEL/',
#                     kb_file_path = 'kb/snomedct_kb.json'
#                    )

# convert_corpus('ncbi-disease/dev_grag.json', out_file='data/ncbi_dev.json')

# def check_train_ent_appers_in_test(train_file, test_file):
#     with open(train_file, 'r') as f:
#         train = json.load(f)
#     train_ents = {}
#     for td in train:
#         train_ents[td['mention_data']['kb_id']] = td

#     count_appers = 0
#     count_didnt_appers = 0

#     with open(test_file, 'r') as f:
#         test = json.load(f)
#     for td in test:
#         test_id = td['mention_data']['kb_id']
#         if test_id in train_ents:
#             count_appers+=1
#         else:
#             count_didnt_appers+=1


#     print(f'train : {len(train)}\ntest : {len(test)}\n{count_appers} GT entities from test set is also appears as GT in train set')


# onto = 'ncbi'
# splitname = 'test'
# corpus_dir = 'ncbi-disease'
# kbpath = 'kb/mesh_kb.json'
# convert_all(onto, splitname, corpus_dir, kbpath)
# filepath = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{corpus_dir}/train_grag_generated_ho.json'
# convert_prime_data_to_train_format(filepath, f'kb/ncbi_prime_train.json', 'kb/mesh_kb.json')
# filepath = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{corpus_dir}/test_grag_generated_ho.json'
# convert_prime_data_to_train_format(filepath, f'kb/ncbi_prime_test.json', 'kb/mesh_kb.json')
# show_len(f"kb/ncbi_prime_test.json")
# show_len(f"kb/ncbi_prime_train.json")
# show_len(f"data/ncbi_train_ho.json")
# check_train_ent_appers_in_test(f'data/ncbi_train_ho.json', 'data/ncbi_test.json')
# check_train_ent_appers_in_test(f'data/ncbi_train_ho.json', 'data/ncbi_test.json')
# check_train_ent_appers_in_test(f'data/ncbi_test_ho.json', 'data/ncbi_test.json')
# data = 'data/blink/zshel/'
# res_file = f'{data}ncbi_test_ho.json'
# res_shuffle_candidates(res_file)

# prepare_for_prime_def(
#     old_prime_file=f'kb/ncbi_prime_{splitname}.json', 
#     new_prime_file=f'kb/ncbi_prime_test_newly_generated.json', 
#     train_data_file = f"data/blink/zshel//{onto}_{splitname}_ho.json"
#     )


# corpus_name = 'bc5cdr'
# splitname = 'train'
# corpus_dir = 'bc5cdr'
# kbpath = 'kb/mesh_kb.json'
# convert_all(corpus_name, splitname, corpus_dir, kbpath)
# show_len(f'kb/{corpus_name}_prime_{splitname}.json')
# show_len(f"data/{corpus_name}_{splitname}_ho.json")
# check_train_ent_appers_in_test(f'data/{corpus_dir}_train_ho.json', f'data/{corpus_dir}_test.json')
# data = 'data/blink/zshel/'
# res_file = f'{data}{corpus_name}_test_ho.json'
# res_shuffle_candidates(res_file)

# onto = 'ncbi'
# splitname = 'train'
# corpus_dir = 'ncbi-disease'
# kbpath = 'kb/mesh_kb.json'
# convert_all(onto, splitname, corpus_dir, kbpath)

# print("_________bc5_____________")

# filepath = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/bc5cdr/train_grag_generated_ho.json'
# convert_prime_data_to_train_format(filepath, f'kb/bc5cdr_prime_train.json', 'kb/mesh_kb.json')
# summary_for_test_ent_appears_in_train(f'data/bc5cdr_train_ho.json', 'data/bc5cdr_test.json')
# filepath = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/bc5cdr/test_grag_generated_ho.json'
# convert_prime_data_to_train_format(filepath, f'kb/bc5cdr_prime_test.json', 'kb/mesh_kb.json')
# summary_for_test_ent_appears_in_train(f'data/bc5cdr_test_ho.json', 'data/bc5cdr_test.json')
# show_len(f'kb/bc5cdr_prime_train.json')
# show_len(f'kb/bc5cdr_prime_test.json')


# print("_________cmo_____________")
# onto = 'cmo'
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_test_ho.json', f"data/{onto}_test_ho.json")
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_test.json', f"data/{onto}_test.json")
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_kb.json', f"kb/{onto}_kb.json")
# filepath = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_grag_generated_ho.json'
# convert_prime_data_to_train_format(filepath, f'kb/{onto}_prime_test.json', f"kb/{onto}_kb.json")
# show_len(f"data/{onto}_test_ho.json")
# show_len(f"data/{onto}_test.json")
# show_len(f"kb/{onto}_kb.json")
# show_len(f'kb/{onto}_prime_test.json')


# print("_________vt_____________")
# onto = 'vt'
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_test_ho.json', f"data/{onto}_test_ho.json")
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_test.json', f"data/{onto}_test.json")
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_kb.json', f"kb/{onto}_kb.json")
# filepath = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_grag_generated_ho.json'
# convert_prime_data_to_train_format(filepath, f'kb/{onto}_prime_test.json', f"kb/{onto}_kb.json")
# show_len(f"data/{onto}_test_ho.json")
# show_len(f"data/{onto}_test.json")
# show_len(f"kb/{onto}_kb.json")
# show_len(f'kb/{onto}_prime_test.json')

# print("_________lpt_____________")
# onto = 'lpt'
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_test_ho.json', f"data/{onto}_test_ho.json")
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_test.json', f"data/{onto}_test.json")
# shutil.copy(f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_kb.json', f"kb/{onto}_kb.json")
# filepath = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/{onto}/{onto}_grag_generated_ho.json'
# convert_prime_data_to_train_format(filepath, f'kb/{onto}_prime_test.json', f"kb/{onto}_kb.json")
# show_len(f"data/{onto}_test_ho.json")
# show_len(f"data/{onto}_test.json")
# show_len(f"kb/{onto}_kb.json")
# show_len(f'kb/{onto}_prime_test.json')

# print("_________NCBI_____________") 
# onto = 'ncbi'
# split_name = 'test'
# outfile = f"data/blink/zshel_64_context_length/{split_name}/{onto}_{split_name}.json"
# kb_file = f"kb/mesh_kb.json"
# prime_kb = f"data/{onto}/blink_format/{split_name}/{onto}_prime_{split_name}_newly_generated.json"
# retri_and_rerank_file = f'models/{onto}/crossencoder_before_fine_tune_crossenc/{split_name}/crossencoder_predictions_grag.json'
# blink_retriver_to_res(onto, split_name, retri_and_rerank_file, kb_file, outfile, prime_kb)

split_name = 'train'
onto = 'ncbi'
create_data_from_blink(onto,split_name)


split_name = 'train'
onto = 'bc5cdr'
create_data_from_blink(onto,split_name)

split_name = 'train'
onto = 'cmo'
create_data_from_blink(onto,split_name)

split_name = 'train'
onto = 'vt'
create_data_from_blink(onto,split_name)


split_name = 'train'
onto = 'lpt'
create_data_from_blink(onto,split_name)

# out_dir = f"data/blink/zshel_64_context_length/{onto}/{split_name}/"
# out_dataset = 'test'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/original_def/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/original_def/top64_candidates/{out_dataset}.json'
# kb_file = f"kb/mesh_kb.json"
# blink_retriver_to_res('original_def', onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, is_prime=False)


# outfile = f"data/blink/zshel_64_context_length/{split_name}/{onto}_{split_name}.json"
# kb_file = f"kb/mesh_kb.json"
# prime_kb = f"data/{onto}/blink_format/{split_name}/{onto}_prime_{split_name}_newly_generated.json"
# retri_and_rerank_file = f'models/{onto}/crossencoder_before_fine_tune_crossenc/{split_name}/crossencoder_predictions_grag.json'
# blink_retriver_to_res(onto, split_name, retri_and_rerank_file, kb_file, outfile, prime_kb)
# onto = 'ncbi'
# split_name = 'train'
# exp = 'prime'
# kb_file = f"kb/mesh_kb.json"
# out_dir = f"data/blink/zshel_64_context_length/{onto}/{split_name}/{exp}/"
# os.makedirs(out_dir, exist_ok=True)
# out_dataset = 'train'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path)
# out_dataset = 'test'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path,
#                       is_prime=False)

# exp = 'ho_prime_others_not'
# kb_file = f"kb/mesh_kb.json"
# out_dir = f"data/blink/zshel_64_context_length/{onto}/{split_name}/{exp}/"
# os.makedirs(out_dir, exist_ok=True)
# out_dataset = 'train'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, 
#                       blink_raw_data_path, ho_prime_others_not=True)


# exp = 'original_def_ho_and_m3_pseudo_pair_for_others_cat'
# kb_file = f"kb/mesh_kb.json"
# out_dir = f"data/blink/zshel_64_context_length/{onto}/{split_name}/{exp}/"
# os.makedirs(out_dir, exist_ok=True)
# out_dataset = 'train'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, 
#                       blink_raw_data_path, ho_prime_others_not=True)


# exp = '(m3_e1)'
# kb_file = f"kb/mesh_kb.json"
# out_dir = f"data/blink/zshel_64_context_length/{onto}/{split_name}/{exp}/"
# os.makedirs(out_dir, exist_ok=True)
# out_dataset = 'train'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(exp, onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, 
#                       blink_raw_data_path, ho_prime_others_not=True)



# out_dataset = 'test'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path,
#                       is_prime=False)

# exp = 'original_def_small_set'
# kb_file = f"kb/mesh_kb.json"
# out_dir = f"data/blink/zshel_64_context_length/{onto}/{split_name}/{exp}/"
# os.makedirs(out_dir, exist_ok=True)
# out_dataset = 'train'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, is_prime=False)
# out_dataset = 'test'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path,
#                       is_prime=False)

# exp = 'original_def'
# kb_file = f"kb/mesh_kb.json"
# out_dir = f"data/blink/zshel_64_context_length/{onto}/{split_name}/{exp}/"
# os.makedirs(out_dir, exist_ok=True)
# out_dataset = 'train'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path, is_prime=False)
# out_dataset = 'test'
# outfile = f"{out_dir}{out_dataset}.json"
# blink_raw_data_path = f"data/{onto}/blink_format/{split_name}/{exp}/"
# blink_retriver_file = f'models/{onto}/biencoder/{split_name}/{exp}/top64_candidates/{out_dataset}.json'
# blink_retriver_to_res(onto, split_name, out_dataset, blink_retriver_file, kb_file, outfile, blink_raw_data_path,
#                       is_prime=False)


# # print("_________BC5CDR_____________") 
# onto = 'bc5cdr'
# split_name = 'test'
# outfile = f"data/blink/zshel_64_context_length/{split_name}/{onto}_{split_name}.json"
# kb_file = f"kb/mesh_kb.json"
# prime_kb = f"data/{onto}/blink_format/{split_name}/{onto}_prime_{split_name}_newly_generated.json"
# retri_and_rerank_file = f'models/{onto}/crossencoder/{split_name}/crossencoder_predictions_grag.json'
# blink_retriver_to_res(onto, split_name, retri_and_rerank_file, kb_file, outfile, prime_kb)





# merge_train_test(onto)
# existig_mc = f"data/{onto}_test_ho.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/bm25/{onto}_test_ho.json")
# existig_mc = f"data/{onto}_test.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/bm25/{onto}_test.json")

# existig_mc = f"data/{onto}_test_ho.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/sentence_similarity/{onto}_test_ho.json", bm25=False)

# existig_mc = f"data/{onto}_test.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/sentence_similarity/{onto}_test.json", bm25=False)

# print("_________BC5CDR_____________")

# onto = 'bc5cdr'
# split_name = 'test'
# outfile = f"data/blink/zshel/{onto}_test.json"
# kb_file = f"kb/mesh_kb.json"
# prime_kb = f"kb/{onto}_prime_{split_name}_defi.json"
# retri_and_rerank_file = f'output/{onto}/zshel_trained/predictions_retri_and_rerank.json'
# blink_retriver_to_res(onto, retri_and_rerank_file, kb_file, outfile, prime_kb)


# outfile = f"data/blink/zshel/{onto}_test.json"
# kb_file = f"kb/mesh_kb.json"
# retri_and_rerank_file = f'output/{onto}/zshel_tarined/predictions_grag.json'
# # blink_retriver_to_res(onto, retri_and_rerank_file, kb_file, outfile)
# prime_kbpath = f"kb/{onto}_prime_{split_name}.json"
# remove_sample_if_gt_not_in_prime(outfile.replace('_test.json', '_test_ho.json'), prime_kbpath)
# prepare_for_prime_def(
#     old_prime_file=f'kb/{onto}_prime_{split_name}.json', 
#     new_prime_file=f'kb/{onto}_prime_test_newly_generated.json', 
#     train_data_file = f"data/blink/zshel//{onto}_{split_name}_ho.json"
#     )

# outfile = f"data/{onto}_train_ho.json"
# kb_file = f"kb/mesh_kb.json"
# blink_retriver_to_res(onto, kb_file, outfile)
# show_len(outfile)
# merge_train_test(onto)
# existig_mc = f"data/{onto}_test_ho.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/bm25/{onto}_test_ho.json")
# existig_mc = f"data/{onto}_test.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/bm25/{onto}_test.json")

# existig_mc = f"data/{onto}_test_ho.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/sentence_similarity/{onto}_test_ho.json", bm25=False)
# existig_mc = f"data/{onto}_test.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/sentence_similarity/{onto}_test.json", bm25=False)

# print("_________CMO_____________") 
# onto = 'cmo'
# split_name = 'test'
# outfile = f"data/blink/blink_ret_fine_tuned/{onto}_test.json"
# kb_file = f"kb/{onto}_kb.json"
# prime_kb = f"kb/{onto}_prime_{split_name}.json"
# retri_and_rerank_file = f'output/{onto}/zshel_trained/predictions_retri_and_rerank.json'
# blink_retriver_to_res(onto, retri_and_rerank_file, kb_file, outfile, prime_kb, made_from_promtel=False)

# print("_________VT_____________") 
# onto = 'vt'
# split_name = 'test'
# outfile = f"data/blink/blink_ret_fine_tuned/{onto}_test.json"
# kb_file = f"kb/{onto}_kb.json"
# prime_kb = f"kb/{onto}_prime_{split_name}.json"
# retri_and_rerank_file = f'output/{onto}/zshel_trained/predictions_retri_and_rerank.json'
# blink_retriver_to_res(onto, retri_and_rerank_file, kb_file, outfile, prime_kb, made_from_promtel=False)

# print("_________LPT_____________") 
# onto = 'lpt'
# split_name = 'test'
# outfile = f"data/blink/blink_ret_fine_tuned/{onto}_test.json"
# kb_file = f"kb/{onto}_kb.json"
# prime_kb = f"kb/{onto}_prime_{split_name}.json"
# retri_and_rerank_file = f'output/{onto}/zshel_trained/predictions_retri_and_rerank.json'
# blink_retriver_to_res(onto, retri_and_rerank_file, kb_file, outfile, prime_kb, made_from_promtel=False)



# merge_train_test(onto)
# existig_mc = f"data/{onto}_test_ho.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/bm25/{onto}_test_ho.json")
# existig_mc = f"data/{onto}_test.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/bm25/{onto}_test.json")

# existig_mc = f"data/{onto}_test_ho.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/sentence_similarity/{onto}_test_ho.json", bm25=False)

# existig_mc = f"data/{onto}_test.json"
# create_data_using_pairwise_retriever(kb_file, existig_mc, f"data/sentence_similarity/{onto}_test.json", bm25=False)