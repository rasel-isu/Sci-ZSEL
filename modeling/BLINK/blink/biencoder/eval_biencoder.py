# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from copy import deepcopy
import json
import logging
import os
from category_eval import MultiGTEvaluation, biencoder_eval_report
import torch
from tqdm import tqdm
import io
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# rasel
# from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizer
# rasel

from blink.biencoder.biencoder import BiEncoderRanker
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel, Stats
from blink.common.params import BlinkParser



def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    logger.info(json.dumps(params, indent=1))

    # Init model 
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    device = reranker.device
    
    cand_encode_path = params.get("cand_encode_path", None)
    
    # candidate encoding is not pre-computed. 
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    
    candidate_pool = data.load_or_generate_candidate_pool(
        tokenizer,
        params,
        logger,
        cand_pool_path,
    )       
    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
            if args.only_get_ent_encoding:
                print(f'only_get_ent_encoding : {args.only_get_ent_encoding}')
                print('No need to generate, already has the encoding!')
                exit()
        except Exception as e:
            logger.info(e)
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = data.encode_candidate(
            reranker,
            candidate_pool,
            params["encode_batch_size"],
            silent=params["silent"],
            logger=logger,
            is_zeshel=params.get("zeshel", None)
            
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)
        
        print(f'args.only_get_ent_encoding {args.only_get_ent_encoding}')
        # input('args.only_get_ent_encoding')

        if args.only_get_ent_encoding:
            exit()  # TODO only for generating candidate encoding
            print(f'only_get_ent_encoding : {args.only_get_ent_encoding}')
        
    test_samples = utils.read_dataset(params["mode"], params["data_path"])
    # test_samples = test_samples[:5]
    logger.info("Read %d test samples." % len(test_samples))

    # rasel : 
    removed = False
    for s in test_samples:
        if 'onto' in s:
            s['world'] = s['onto']
            del s['onto']
            removed = True
        # print(s)
        # input('s')
    if removed:
        print(f'There was a key named "onto", whcih is removed and "world" is set as onto')
    # rasel : 
   

    test_data, test_tensor_data, samples_map = data.process_mention_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params['context_key'],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, 
        sampler=test_sampler, 
        batch_size=params["eval_batch_size"]
    )
    
    # rasel
    with open(f'{params["kb_file_path"]}') as f:
        exact_kb = json.load(f)
        if params["onto"] == 'ncbi':
            kb = {}
            for e in exact_kb:
                ent = exact_kb[e]
                kb[e] = ent
                for i in ent['altdiseaseid']:
                    kb[i] = ent
        elif params["onto"] == 'bc5cdr':
            kb = deepcopy(exact_kb)
        elif params["onto"] in ['cmo', 'vt', 'lpt']:
            kb = deepcopy(exact_kb)


    with open(f'{params["data_path"]}/id_map.json') as f:
        map_dict = json.load(f)
        swapped_map_dict = {v: k for k, v in map_dict.items()}

    with open(f'{params["grag_data_path"]}/{params["mode"]}_grag.json') as f:
        raw_grag_data = json.load(f)
        grag_data={}
        for i in raw_grag_data:
            gt_list = i['ground_truth']
            gt_with_mapped_id = []
            for gt in gt_list:
                if gt['id'] in swapped_map_dict:
                    gt['map_id'] = int(swapped_map_dict[gt['id']])
                else:
                    gt['map_id'] = None

                gt_with_mapped_id.append(gt)
            i['ground_truth'] = gt_with_mapped_id
            grag_data[i['sample_id']] = i

    # rasel
    save_results = params.get("save_topk_result")
    new_data, pred_rakings = nnquery.get_topk_predictions(
        reranker,
        test_dataloader,
        candidate_pool,
        candidate_encoding,
        samples_map,
        grag_data,
        map_dict,
        exact_kb,
        params,
        params["silent"],
        logger,
        params["top_k"],
        params.get("zeshel", None),
        save_results,
    )

    if save_results: 
        save_data_dir = os.path.join(
            params['output_path'],
            "top%d_candidates" % params['top_k'],
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % params['mode'])
        torch.save(new_data, save_data_path)

        # rasel
        bi_pred_file = save_data_path.replace('.t7', '.json')
        with open(bi_pred_file, 'w') as f:
            json.dump(pred_rakings, f, indent=1)

        biencoder_eval_report(params,bi_pred_file)
        


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__

    mode_list = params["mode"].split(',')
    for mode in mode_list:
        new_params = params
        new_params["mode"] = mode
        main(new_params)
