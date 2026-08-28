import io
import json

from category_eval import Evaluation, MultiGTEvaluation

def add_space_to_context(txt):
    pass

def save_report(onto, id_map, pfile):
    with open(pfile) as f:
        pred = json.load(f)
    with open(f'{id_map}', 'r') as f:
        id_map = json.load(f)

    converted = []
    for inc, p in enumerate(pred):
        m = p['test_data'][0]
        
        b = p['bi'][0]['prediction']
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

        # mention_context = m['context_left'] + '[MENTION_START]'+mention+'[MENTION_END]'+m['context_right']
        
        unique_triple = {}
        for bitem in b:
            # eid = bitem['id']
            eid = id_map[bitem['id']]
            unique_triple[eid]=bitem
        candidates = p['cross'][0]

        cnd_id = []
        retrieved_candidates = []
        for c in candidates:
            cnd_id.append(c['id'])
            c['id'] = id_map[c['id']]
            retrieved_candidates.append(c)
            
        
        # with open(f'models/ncbi/biencoder/nn_predictions.json', 'r') as f:
        #     nnp = json.load(f)
        # print(nnp[inc])
        # # print(cnd_id)


        d = {'mention_id' : '111111',
            'mention': mention.strip(),
            'mention_context':mention_context.strip(),
            'ground_truth': {'id':id_map[str(m['label_id'])], 'title':m['label_title']},
            'retriever_result_gt':p['bi'][0]['gt_score'],
            'unique_triple':unique_triple,
            'retrieved_candidates':retrieved_candidates
            }
        converted.append(d)
    
    with open(f"{pfile.replace('.json', '_grag.json')}", 'w') as f:
        json.dump(converted, f, indent=1)
    
    eval_bi = Evaluation(converted, for_retrieval=True)
    not_none_data, none_data = eval_bi.get_report()

    eval_cross = Evaluation(converted, for_retrieval=False)
    not_none_data, none_data = eval_cross.get_report()

    

    report = f'Bi-Encoder\n{"_"*20}\n{eval_bi.text_report}\n\nCross-Encoder\n{"_"*20}\n{eval_cross.text_report}'


    with open(f"{pfile.replace('.json', '_eval.txt')}", 'w') as f:
        f.write(report)


    with open(f"{pfile.replace('.json', '_category_info.json')}", 'w') as f:
        json.dump(eval_cross.cat_wise_gt_matched, f, indent=1)

    with open(f"{pfile.replace('.json', '_retri_and_rerank.json')}", 'w') as f:
        json.dump(converted, f, indent=1)

def eval_grag(file):
    with open(file) as f:
        pred = json.load(f)
    eval_bi = Evaluation(pred, for_retrieval=True)
    not_none_data, none_data = eval_bi.get_report()
    print(eval_bi.text_report)

def evaluate_biencoder(data_dir, pfile):
    with open(pfile) as f:
        all_predictions = json.load(f)
    with open(f'{data_dir}/id_map.json', 'r') as f:
        id_map = json.load(f)

    kb_dict = {}
    with io.open(f'{data_dir}/kb.jsonl', mode="r", encoding="utf-8") as file:
        for line in file:
            e = json.loads(line.strip())
            kb_dict[e['id']]=e

    converted = []
    for predictions in all_predictions:
        for men_pred in predictions:
            
            retrieved_candidates = []
            unique_triple = {}
            for c in men_pred['retriever_predictions']:
                c['title'] = kb_dict[c['id']]['title']
                c['id'] = id_map[str(c['id'])]
                unique_triple[c['id']] = c
                retrieved_candidates.append(c)

            m = men_pred['mention_data']
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


            d = {'mention_id' : '111111',
            'mention': mention,
            'mention_context':mention_context.strip(),
            'ground_truth': {'id':id_map[str(m['label_id'])], 'title':m['label_title']},
            'retriever_result_gt':men_pred['retriever_retrived_gt'],
            'retrieved_candidates':retrieved_candidates,
            'unique_triple':unique_triple
            }
        converted.append(d)

    with open(f"{pfile.replace('.json', '_grag.json')}", 'w') as f:
        json.dump(converted, f, indent=1)

    eval_bi = Evaluation(converted, for_retrieval=True)
    not_none_data, none_data = eval_bi.get_report()

    report = f'Bi-Encoder\n{"_"*20}\n{eval_bi.text_report}\n\nCross-Encoder\n{"_"*20}'

    with open(f"{pfile.replace('.json', '_eval.txt')}", 'w') as f:
        f.write(report)

    with open(f"{pfile.replace('.json', '_category_info.json')}", 'w') as f:
        json.dump(eval_bi.cat_wise_gt_matched, f, indent=1)

def get_cand_info(cross_pred, kb_dict, id_map):
    reranker_predictions = cross_pred['sorted_cand_score']
    reranker_cands = []
    for c in reranker_predictions:
        c['title'] = kb_dict[c['id']]['name']
        c['id'] = c['id']
        reranker_cands.append(c)
    return reranker_cands

def get_reranker_cands(bi_pred, cross_pred, kb_dict, id_map):
    return get_cand_info(cross_pred, kb_dict, id_map)


def evaluate_biencoder_and_crossencoder(params, bi_pfile, cross_pfile, split_name='test'):
    data_dir = params["raw_data_path"]

    with open(f'{params["grag_data_path"]}/{split_name}_grag.json') as f:
        multiple_gt_grag = json.load(f)
        multiple_gt_grag_dict = {}
        for item in multiple_gt_grag:
            multiple_gt_grag_dict[item['sample_id']] = item

    with open(f'{params["kb_file_path"]}') as f:
        exact_kb = json.load(f)

    with open(bi_pfile) as f:
        bi_all_predictions = json.load(f)
    with open(cross_pfile) as f:
        cross_all_predictions = json.load(f)

    with open(f'{data_dir}id_map.json', 'r') as f:
        id_map = json.load(f)

    kb_dict = {}
    with io.open(f'{data_dir}kb.jsonl', mode="r", encoding="utf-8") as file:
        for line in file:
            e = json.loads(line.strip())
            kb_dict[e['id']]=e

    converted = []
    if not len(bi_all_predictions)==len(cross_all_predictions):
        raise ValueError(f'Bi={len(bi_all_predictions)}, cross={len(cross_all_predictions)}, preds are not same!')
         
    for bi_pred, cross_pred in zip(bi_all_predictions, cross_all_predictions):
        
        # Compare sample ID from both
        bi_pred_sample_id = bi_pred['mention_data']['sample_id']
        cross_pred_sample_id = int(cross_pred['reranked_for'])
        if bi_pred_sample_id != cross_pred_sample_id:
            both_id_text = f'Bi-encoder ID : {bi_pred_sample_id}, Cross-encoder ID : {cross_pred_sample_id}'
            raise ValueError(f'Bi-encoder sample and Cross-encoder are not same!\n{both_id_text}')
        
        # for bi_pred, cross_pred  in zip(bi_predictions, cross_predictions):
        
        reranked_candidates = get_cand_info(cross_pred, exact_kb, id_map)
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
        'reranker_result_gt':cross_pred['reranker_gt_rank'],
        'reranked_candidates':reranked_candidates,
        'retriever_predictions':retriever_predictions
        }
        converted.append(d)

    with open(f"{cross_pfile.replace('.json', '_grag.json')}", 'w') as f:
        json.dump(converted, f, indent=1)

    eval_bi = MultiGTEvaluation(converted, exact_kb, 'retriever_predictions', multiple_gt_grag, for_retrieval=True)
    not_none_data, none_data = eval_bi.get_report()

    eval_cross = MultiGTEvaluation(converted, exact_kb, 'reranked_candidates', multiple_gt_grag, for_retrieval=False)
    not_none_data, none_data = eval_cross.get_report()

    report = f'Bi-Encoder\n{"_"*20}\n{eval_bi.text_report}\n{"_"*20}\n\nCross-Encoder\n{"_"*20}\n{eval_cross.text_report}'

    with open(f"{cross_pfile.replace('.json', '_eval.txt')}", 'w') as f:
        f.write(report)

    with open(f"{cross_pfile.replace('.json', '_cross_category_info.json')}", 'w') as f:
        json.dump(eval_cross.cat_wise_gt_matched, f, indent=1)

    return {
        'eval_cross':{'recall_at_1':eval_cross.recall_at_1}
    }

