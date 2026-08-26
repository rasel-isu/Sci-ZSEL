from eval import ReportMaker

ontologies = [
    'bc5cdr',
    # 'ncbi',
    # 'cmo',
    # 'vt',
    # 'lpt'
]
split_name = 'train'

seeds = [
        0,
        42,
        52313
    ]
    
for seed in seeds:
    for onto in ontologies:
        report_for = 'crossencoder'
        eval_text_name = '_eval.txt'
        # report_start_end_text = ['Overall Accuracy', 'within top-5 candidates']

        report_start_end_text = {'recall@1':['Overall Accuracy', 'top-5 candidates'], 
        'recall@5':['Overall Accuracy', 'top-5 candidates', 'top-64 candidates']} 

        text_eval_file_dir = f'model_disambiguation/{onto}/{split_name}/seed-{seed}/'
        maker = ReportMaker(
            onto=onto,
            report_save_to=text_eval_file_dir,
            report_start_end_text=report_start_end_text,
            firt_row_text='Retriever:BLINK, Re-ranker:Fine-tune reranker ReS',
            before_ft_dir=f'model_disambiguation/{onto}/{split_name}/before_fine_tune/',
            train_data_path = f'data/blink/zshel_64_context_length/{onto}/{split_name}/'
        )
        exps = [
            # 'prime',
            # "(m1_e1)",
            # "(m1_e1)_(m1s==e)_(m1==es)",
            # "(m1_e1)_(m3_e1)",
            # '(m3_e1)',
            #  "(m1_e1)_(m3_e1)_(m3_e1)_from_(m3s==e')_(m3==e's)",
            # '(m4_e2)',
            # "(m4_e2)_(m4s==e')_(m4==e's)",
            # "(m4_e2)_rm_sm_e2",
            # "(m4_e2)_(m4s==e')_(m4==e's)_rm_sm_e2",
            # '(m5_e3)',
            # '(m3_e1)_(m4_e2)_(m5_e3)',
            # '(m1_e1)_(m3_e1)_(m4_e2)_(m5_e3)',
            # "(m1_e1)_(m3_e1)_(m4_e2)",
            # "(m1_e1)_(m3_e1)_(m4_e2)_34_rm_sm_e",
            #  "(m1_e1)_(m3_e1)_(m4_e2)_and_their_plurals",
            # "(m1_e1)_(m3_e1)_(m4_e2)_and_their_plurals_34_rm_sm_e",
            # '(m6_e4)',
            #  "(m1_e1)_(m3_e1)_(m4_e2)_(m5_e3)_(m6_e4)",
            # "(m1_e1)_(m3_e1)_(m4_e2)_(m5_e3)_(m6_e4)_3456_rm_sm_e",
            # "(m1_e1)_(m3_e1)_(m4_e2)_(m5_e3)_(m6_e4)_and_plurals",
            # " ho_prime_theta_70",
            # 'original_def_small_set', 
            # 'original_def', 
            # "original_def_small_set",
            # 'ho_prime_others_not'

            "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime",
            "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e",
            "synonym",
            "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e",
            "synonymU(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e",
            "synonymU(m1_e1)"
            ]
        maker.make_pptx_report_for_all_epoch(text_eval_file_dir,eval_text_name, exps, 3, 3)
