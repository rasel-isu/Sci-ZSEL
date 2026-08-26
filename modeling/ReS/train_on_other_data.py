import argparse
import json
import os
from data_disambiguation import load_data, load_entities
from run_disambiguation_attention import main


def train_exp(data_dir,
              lora,
              corpus_name, 
              onto_name,
              exp, 
              split_name, 
              use_title_during_testing, 
              both_set=False):
            
    seeds = [
            0,
            42,
            52313
        ]
    
    for seed in seeds:
        
    
        f_setting = list(exp.keys())[0]
        parser = argparse.ArgumentParser()
        parser.add_argument("--onto", default=onto_name)
        parser.add_argument("--corpus", default=corpus_name)
        parser.add_argument("--split_name", default=split_name)
        parser.add_argument("--exp", default=f_setting)

        parser.add_argument("--seed", default=seed)
        if lora:
            model_dir = f"model_disambiguation/{corpus_name}/{split_name}/seed-{seed}/{f_setting}/lora/"
        else:
            model_dir = f"model_disambiguation/{corpus_name}/{split_name}/seed-{seed}/{f_setting}/"
        
        model_path = f"{model_dir}{corpus_name}_{f_setting}.pt"
        parser.add_argument("--model", default=model_path)


        num_train_epoch = 3
        if onto_name =='medic':
            kb_file_path = f"{data_dir.replace(f_setting, '')}other_files/only_medic_def.json"
        elif onto_name =='mesh':
            kb_file_path = f"{data_dir.replace(f_setting, '')}other_files/merged_onto.json"
        elif onto_name in ['cmo', 'vt', 'lpt']:
            kb_file_path = f"{data_dir.replace(f_setting, '')}other_files/{onto_name}_kb.json"

        
        parser.add_argument("--grag_data_path", default=f"{data_dir.replace(f_setting, '')}other_files")
        parser.add_argument("--kb_file_path", default=kb_file_path)

        if f_setting == 'original_def_small_set':
            fine_tune_setting = 'original_title'
        elif f_setting == 'original_def':
            fine_tune_setting = 'original_title' 
        elif f_setting == 'ho_prime_others_not':
            fine_tune_setting = 'prime' 
        elif exp[f_setting] in [None, 'm1','m3', 'm4', 'm5', 'm6']:
            fine_tune_setting = 'original_title' 
        else:
            fine_tune_setting = f_setting

        parser.add_argument("--fine_tune_setting", default=fine_tune_setting)
        parser.add_argument("--fine_tune", default=True)


        parser.add_argument("--epochs", default=num_train_epoch)
        parser.add_argument("--lora", default=lora)
        
        # rasel : small
        # 5e-5
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--batch", default=8,type=int)
        # rasel : prime kb
        if both_set:
            parser.add_argument("--train_kb_prime", default=f"kb/{onto_name}_prime_train_and_test.json")
            parser.add_argument("--train_data", default=f"{data_dir}{corpus_name}_train_and_test_ho.json")
        else:
            if fine_tune_setting == 'prime':
                if exp[f_setting] =='m3':
                    prime_kb_file = f"{corpus_name}_prime_{split_name}_newly_generated_prime.json"
                elif exp[f_setting] =='m4':
                    prime_kb_file = f"top_1_from_biencoder_newly_generated_prime.json"
                elif exp[f_setting] =='m5':
                    prime_kb_file = f"e3_from_parent_child_of_e1_newly_generated_prime.json"
                elif exp[f_setting] =='m6':
                    prime_kb_file = f"mention_overlap_unq_terms_newly_generated_prime.json"
                parser.add_argument("--train_kb_prime", default=f"kb/{prime_kb_file}")
            else:
                parser.add_argument("--train_kb_prime", default=None)
            
            # train_data_filename = f"{data_dir}{corpus_name}_{split_name}_ho_shuffled_candidates.json"
            # train_data_filename = f"{data_dir}{corpus_name}_{split_name}_ho.json"

            if split_name=='train':
                train_data_filename = f"{data_dir}train.json"

            parser.add_argument("--train_data", default=train_data_filename)

        test_data_filepath = data_dir.replace(f_setting, 'original_def_prmbl')
        parser.add_argument("--train_kb", default=f"kb/{onto_name}_kb.json")
        parser.add_argument("--dev_kb", default=f"kb/{onto_name}_kb.json")
        parser.add_argument("--dev_data", default=f"{test_data_filepath}test.json")

        
        parser.add_argument("--use_title_during_testing", default=use_title_during_testing)

        parser.add_argument("--pred_data", default=None)
        parser.add_argument("--eval_before_fine_tune", default=False)


        # rasel : load trained model
        parser.add_argument("--saved_pt_model",
                            default="model_disambiguation/zeshel_disambiguation_attention.pt")

        parser.add_argument("--transformer_model",
                            default="roberta-base")
        
        # parser.add_argument("--transformer_model",
        #                     default="allenai/biomed_roberta_base")

        parser.add_argument("--cand_num_train", default=21,type=int)
        parser.add_argument("--cand_num", default=64,type=int)

        parser.add_argument("--type_loss", type=str,
                            default="sum_log_nce",
                            choices=["log_sum", "sum_log", "sum_log_nce",
                                        "max_min", "bce_loss"])
        parser.add_argument("--max_len", default=512, type=int)
        parser.add_argument("--max_ent_len", default=256, type=int)
        parser.add_argument("--max_text_len", default=256, type=int)

        parser.add_argument("--forgot_test_data", default=f"{test_data_filepath}test.json")
        parser.add_argument("--lego_test_data", default=f"{test_data_filepath}test.json")
        parser.add_argument("--star_test_data", default=f"{test_data_filepath}test.json")
        parser.add_argument("--yugioh_test_data", default=f"{test_data_filepath}test.json")


        parser.add_argument("--forgot_kb", default=f"kb/{onto_name}_kb.json")
        parser.add_argument("--lego_kb", default=f"kb/{onto_name}_kb.json")
        parser.add_argument("--star_kb", default=f"kb/{onto_name}_kb.json")
        parser.add_argument("--yugioh_kb", default=f"kb/{onto_name}_kb.json")


        parser.add_argument("--warmup_proportion", default=0.1)
        parser.add_argument("--weight_decay", default=0.01)
        parser.add_argument("--adam_epsilon", default=1e-6, type=float)
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
        
        parser.add_argument("--num_workers", default=0)
        parser.add_argument("--simpleoptim", default=False)
        parser.add_argument("--clip", default=1)
        parser.add_argument("--info_token_num", default=3, type=int)
        parser.add_argument("--gpus", default="0,1,2,3")
        parser.add_argument("--logging_steps", default=100)
        parser.add_argument("--eval_step", default=10000, type=int)
        parser.add_argument("--do_train", action="store_true", default=True)
        parser.add_argument("--do_eval", action="store_true", default=False)
        parser.add_argument("--do_eval_only_each_epoch", action="store_true", default=False)
        # parser.add_argument("--do_eval_only_each_epoch", action="store_true", default=True)

        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus



        main(args)


