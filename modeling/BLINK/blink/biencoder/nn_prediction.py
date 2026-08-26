# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import re
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats
from utils import MEDICGraph, MESHGraph, get_connected_ents_for_the_label

def get_gt_score_and_ranks_for_test_set(samples_map, grag_data, map_dict, kb, sample_ids,
                           scores, label_ids, top_k):
    scores_all, pred_indicies_all = scores.topk(top_k)
    scores_all = scores_all.tolist()
    pred_indicies_all = pred_indicies_all.tolist()
    label_ids_all = label_ids.tolist()
    sample_all = sample_ids.tolist()
    all_gt = []
    mention_data = []
    for scr, pred_indx, lbl_id, smpl_id in zip(scores_all, pred_indicies_all, label_ids_all, sample_all):
        
        gt_list = grag_data[smpl_id]['ground_truth']
        ranking = []
        gt_score = []
        rank_num = 1
        for scr_i, pred_indx_i in zip(scr, pred_indx):
            
            
            # check multiple GT
            for gt_item in gt_list:
                if gt_item['map_id'] == pred_indx_i:
                    the_gt = {'gt_id':gt_item['id'], "gt_title":gt_item['title'],'map_id':gt_item['map_id']}
                    gt_score.append({** the_gt, **{'candt':map_dict[str(pred_indx_i)],'score':scr_i, 'rank':rank_num}}) 

            # check altids
            pred_indx_to_kb_id = map_dict[str(pred_indx_i)]
            if 'altdiseaseid' in kb[pred_indx_to_kb_id]:
                altids = kb[pred_indx_to_kb_id]['altdiseaseid']
                for alid in altids:
                    for gt_item in gt_list:
                        if gt_item['id'] == alid:
                            the_gt = {'gt_id':gt_item['id'], "gt_title":gt_item['title'],'map_id':gt_item['map_id']}
                            gt_score.append({**the_gt, **{'candt':pred_indx_to_kb_id, 'altdiseaseid':alid, 'score':scr_i, 'rank':rank_num}})

            
            ranking.append({'id':pred_indx_i, 'score':scr_i})
            rank_num+=1

        all_gt.append(gt_score)
        d = {
            'mention_data':samples_map[smpl_id],
            'retriever_retrived_gt':gt_score,
            'retriever_predictions':ranking
        }
        mention_data.append(d)

        
    return all_gt, mention_data

def get_gt_score_and_ranks_for_train_set(samples_map, grag_data, map_dict, kb, sample_ids,
                           scores, label_ids, top_k):
    scores_all, pred_indicies_all = scores.topk(top_k)
    scores_all = scores_all.tolist()
    pred_indicies_all = pred_indicies_all.tolist()
    label_ids_all = label_ids.tolist()
    sample_all = sample_ids.tolist()
    all_gt = []
    mention_data = []
    for scr, pred_indx, lbl_id, smpl_id in zip(scores_all, pred_indicies_all, label_ids_all, sample_all):
        
        gt_list = grag_data[smpl_id]['ground_truth']
        ranking = []
        gt_score = {}
        rank_num = 1
        for scr_i, pred_indx_i in zip(scr, pred_indx):
            
            
            # check multiple GT
            for gt_item in gt_list:
                if gt_item['map_id'] == pred_indx_i:
                    gt_score = {'gt': gt_item, 'score':scr_i, 'rank':rank_num}

            # check altids
            pred_indx_to_kb_id = map_dict[str(pred_indx_i)]
            if 'altdiseaseid' in kb[pred_indx_to_kb_id]:
                altids = kb[pred_indx_to_kb_id]['altdiseaseid']
                for alid in altids:
                    for gt_item in gt_list:
                        if gt_item['id'] == alid:
                            gt_score = {'gt': gt_item, 'score':scr_i, 'rank':rank_num, 'altdiseaseid':alid}

            
            ranking.append({'id':pred_indx_i, 'score':scr_i})
            rank_num+=1

        all_gt.append(gt_score)
        d = {
            'mention_data':samples_map[smpl_id],
            'retriever_retrived_gt':gt_score,
            'retriever_predictions':ranking
        }
        mention_data.append(d)

        
    return all_gt, mention_data

def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    samples_map,
    grag_data,
    map_dict,
    kb,
    params,
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)


    

    nn_context = []
    nn_candidates = []
    connected_candidates_graph = []
    connected_candidates_graph_kb_int_id = []
    connected_labels_graph = []

    nn_labels = []
    nn_worlds = []
    stats = {}
    pred_rakings = []

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)

    # rasel
    matched = 0
    rank_1_count = 0

    if params["onto"] =='bc5cdr':
        graph_obj = MESHGraph()
    else:
        graph_obj = MEDICGraph(params["kb_file_path"])


    with open(f'{params["data_path"]}/id_map.json') as f:
        map_int_to_kb = json.load(f)
        map_kb_to_int = {v: k for k, v in map_dict.items()}
               
    # rasel
    debug_texts = []
    cands = []
    prime_candidate_ids = []
    sample_ids = []
    oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_token_ids, srcs, label_ids, sample_id = batch

        src = srcs[0].item()

        # Biencoder didnt use prime while predicting top k candidates
        scores = reranker.score_candidate(
            context_input, 
            None, 
            cand_encs=cand_encode_list[src].to(device)
        )
        values, indicies = scores.topk(top_k)
        
        # rasel
        if params["mode"] == 'test':
            all_gt, mention_data = get_gt_score_and_ranks_for_test_set(samples_map, grag_data, 
                                                      map_dict, kb, sample_id, 
                                                      scores, label_ids, top_k)
        elif params["mode"] == 'train':
            all_gt, mention_data = get_gt_score_and_ranks_for_train_set(samples_map, grag_data,
                                                      map_dict, kb, sample_id, 
                                                      scores, label_ids, top_k)
            
        pred_rakings.extend(mention_data)
        for gt in all_gt:
            if 'rank' in gt:
                if gt['rank']==1:
                    rank_1_count+=1
        # rasel

        old_src = src
        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i]

            if srcs[i] != old_src:
                print('srcs[i] != old_src')
                src = srcs[i].item()
                # not the same domain, need to re-do
                new_scores = reranker.score_candidate(
                    context_input[[i]], 
                    None,
                    cand_encs=cand_encode_list[src].to(device)
                )
                _, inds = new_scores.topk(top_k)
                inds = inds[0]

            pointer = -1
            for j in range(top_k):
                if inds[j].item() == label_ids[i].item():
                    pointer = j
                    break
            stats[src].add(pointer)

            if pointer == -1:

                # rasel
                pointer = top_k-1
                actual_label_id = label_ids[i]
                inds = torch.cat((inds[:-1], actual_label_id))

                # continue
            else:
                matched+=1

            cands.append(inds.tolist()) 
            # rasel

            if not save_predictions:
                continue
            
            
            # add examples in new_data
            # cur_candidates = candidate_pool[src][inds]
            cur_candidates = candidate_pool[src][inds.cpu()] # rasel
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_worlds.append(src)

            # rasel
            if params['has_gt']:
                actual_label_id = label_ids[i].item()
                connected_inds = get_connected_ents_for_the_label(graph_obj, map_int_to_kb, map_kb_to_int, 
                                                                actual_label_id)
                
                connected_labels_graph.append(0) # label inx is 0 because actual_label_id adeed at the 0 position : see next line
                connected_inds = torch.tensor([actual_label_id] + connected_inds)
                connected_candidates_graph_kb_int_id.append(connected_inds.tolist())
                cur_conn_candidates = candidate_pool[src][connected_inds.cpu()]
                connected_candidates_graph.append(cur_conn_candidates.cpu().tolist())

                # conn_candidate_decoded = ''
                # for item in cur_conn_candidates.cpu().tolist():
                #     conn_candidate_decoded += f'{reranker.tokenizer.decode(item, skip_special_tokens=False)}\n'
                    

            original = cur_candidates.cpu().tolist()[pointer]
            original_decoded = reranker.tokenizer.decode(original, skip_special_tokens=False)
            all_cands = cur_candidates.cpu().tolist()
            all_cands[pointer]=candidate_token_ids[i].cpu().tolist()
            prime =  all_cands[pointer]
            prime_decoded = reranker.tokenizer.decode(prime, skip_special_tokens=False)
            prime_candidate_ids.append(all_cands)
            sample_ids.append(sample_id[i])

            context_data = context_input[i].cpu().tolist()
            context_decoded = reranker.tokenizer.decode(context_data, skip_special_tokens=False)
            match = re.search(r"\[unused0\](.*?)\[unused1\]", context_decoded)
            if match:
                mention = match.group(1).strip()
                if mention=='hemochromatosis':
                    debug_texts.append(f'mention : {mention}\nGT: {original_decoded}\n\n\n\n')


                

    # with open('debug.txt', 'w') as f:
    #     f.write('\n\n'.join(debug_texts))
    # rasel

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())


    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    # rasel
    nn_cands_id = torch.LongTensor(cands)
    prime_candidate_ids_tensor = torch.LongTensor(prime_candidate_ids)
    sample_ids_tensor = torch.LongTensor(sample_ids)
    connected_labels_graph = torch.LongTensor(connected_labels_graph)
    connected_candidates_graph = [torch.LongTensor(item) for item in connected_candidates_graph]    
    # rasel
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
        'candidate_kb_integer_ids': nn_cands_id,
        'prime_candidates': prime_candidate_ids_tensor,
        'sample_ids': sample_ids_tensor,
        'connected_labels_graph': connected_labels_graph,
        'connected_candidates_graph': connected_candidates_graph,
        'connected_candidates_graph_kb_int_id': connected_candidates_graph_kb_int_id
    }
    # rasel
    print(f'len {len(train_dataloader)}, matched : {matched}')
    print(f'acc : {matched/len(train_dataloader)}')
    print(f'rank_1_count : {rank_1_count}')
    # rasel

    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data, pred_rakings

