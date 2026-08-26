import json
from category_eval import ReportMaker

def compare_incorrect_prediction(exp_dir, exps, num_sample=881,reacall_at=1):

    key_map = {
        "prime": "prime_________________",
        "original_def_ho": "original_def_ho_______",
        "original_def_ho_m3":'original_def_ho_m3____',
        "ho_prime_theta_70":"ho_prime_theta_70_____",
        "original_def_small_set":"original_def_small_set",
        "original_def": "original_def__________",
        "ho_prime_others_not": "ho_prime_others_not___"
        }

    common_incorrect_prediction_index = {}
    for en in exps:
        print(f'Best epoch dir : {exps[en]} for {en}')
        filepath = f'{exp_dir+en}/{exps[en]}/crossencoder_predictions_grag.json'
        if en not in common_incorrect_prediction_index:
            common_incorrect_prediction_index[en]={}
        with open(filepath) as f:
            predictions = json.load(f)
        if len(predictions)!=num_sample:
            raise ValueError ('Invalid number of samples')
        for i,pred in enumerate(predictions):
            gtid = pred['ground_truth']['id']
            candidate = pred['retrieved_candidates'][reacall_at-1]
            if gtid != candidate['id']:
                common_incorrect_prediction_index[en][i]=pred
    common_incorrect_samples = []
    for i in range(num_sample):
        rankings = {}
        matched = True
        for exp in common_incorrect_prediction_index:
            

            if i not in common_incorrect_prediction_index[exp]:
                matched = False
                break
            reranker_result_gt = common_incorrect_prediction_index[exp][i]['reranker_result_gt']
            rank = {'retriever_result_gt' : common_incorrect_prediction_index[exp][i]['retriever_result_gt'],
                'reranker_result_gt' : {'gt': reranker_result_gt['gt'], 'rank': reranker_result_gt['rank'], 'score': reranker_result_gt['score']}
                }

            rankings[key_map[exp]] = str(rank)

        if matched:

            mdata = common_incorrect_prediction_index[exp][i]
            del mdata['retrieved_candidates']
            del mdata['unique_triple']
            del mdata['retriever_result_gt']
            del mdata['reranker_result_gt']
            mdata['ranking'] = rankings
            common_incorrect_samples.append(mdata)
    
    with open(f'{exp_dir}common_incorrect_prediction.json', 'w') as f:
        json.dump(common_incorrect_samples, f, indent=2)



variants = [
    "BASE-NEG-ENTIRE-KB",
    "remove_prchsbl_from_neg_list",
    "add_prch_in_pos_list",
    ]
# batch_sizes = [
#     128,
#     256,
#     512,
#     ]
seeds=[
  0,
  42,
  52313
]
ontologies = [
    # 'lpt',
    # 'vt',
    # 'cmo',
    # 'ncbi',
    # 'bc5cdr',
    'cometa',
]
sptitname = 'train'
for seed in seeds:
    for variant in variants:
        for onto in ontologies:
            bienc_report_file = 'test_eval.txt'
            bienc_report_start_end_text_for_recall = {'recall@1':['Bi-Encoder', 'top-5 candidates'], 
                                                'recall@5':['Bi-Encoder', 'top-5 candidates', 'top-32 candidates'],
                                                'recall@32':['top-5 candidates', 'top-32 candidates', 'top-64 candidates'],
                                                'recall@64':['top-64 candidates']
                                                } 
            bienc_model_dir = f'/seed-{seed}/fine-tuned-{variant}'
            bienc_report_save_to=f'models/{onto}/biencoder/train/{bienc_model_dir}/'
            bi_text_eval_file_dir = f'models/{onto}/biencoder/{sptitname}/{bienc_model_dir}/'


            cross_report_file = 'crossencoder_predictions_eval.txt'
            report_start_end_text_for_recall = {'recall@1':['Cross-Encoder', 'top-5 candidates'], 
                                                'recall@5':['Cross-Encoder', 'top-5 candidates', 'top-32 candidates']
                                                } 
            cross_report_save_to=f'models/{onto}/crossencoder/train/fine-tune/seed-{seed}'
            maker = ReportMaker(
                onto=onto,
                sptitname = sptitname,

                bi_report_save_to=bienc_report_save_to,
                bi_report_start_end=bienc_report_start_end_text_for_recall,
                bi_firt_row_text='Fine-tune Retriever',
                bi_before_ft_dir=f'models/{onto}/biencoder/{sptitname}/original_def_prmbl/top64_candidates/before_fine_tune/',

                cross_report_save_to=cross_report_save_to,
                cross_report_start_end=report_start_end_text_for_recall,
                cross_firt_row_text='Retriever:BLINK, Re-ranker:Fine-tune reranker',
                cross_before_ft_dir=f'models/{onto}/crossencoder/{sptitname}/fine-tune/original_def_prmbl/before_fine_tune/'
                
                )
            cross_text_eval_file_dir = f'models/{onto}/crossencoder/{sptitname}/fine-tune/seed-{seed}/'
            exps = [
                "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime",
                "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e",
                "synonym",
                "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e",
                
                # "synonymU(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e",
                # "synonymU(m1_e1)",
                ]
            
            # exp_from = '(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e'
            exp_from = None
            
            if onto in ['bc5cdr', 'cometa']:
                last_epoch = 3
            else:
                last_epoch = 0
                
            maker.make_report_for_biencoder(bi_text_eval_file_dir,bienc_report_file, exps, last_epoch, last_epoch,
                                                exp_from=exp_from,
                                                bienc_model_dir=bienc_model_dir)

            
            # maker.make_report_for_crossencoder(cross_text_eval_file_dir, cross_report_file, exps, 0, 2, exp_from=exp_from )



            # compare_incorrect_prediction(report_save_to, maker.best_epoch_dir)
            # compare_incorrect_prediction(report_save_to, {'prime': 'epoch_5', 'original_def_ho': 'epoch_0', 'ho_prime_theta_70': 'epoch_0', 'original_def_small_set': 'epoch_1', 'original_def': 'epoch_6', 'ho_prime_others_not': 'epoch_8'})



