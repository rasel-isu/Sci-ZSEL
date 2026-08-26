import argparse
import json
import os
from data_disambiguation import load_data, load_entities
from run_disambiguation_attention import main



def eval_exp(data_dir, 
             f_setting, 
             lora, 
             corpus_name, 
             exp, 
             split_name, 
             use_title_during_testing, 
             before_fine_tune=False):
    
    fname = 'without_test_title'
    if use_title_during_testing:
        fname = 'with_test_title'
    
    if lora:
        model_dir = f"model_disambiguation/{corpus_name}/{split_name}/{f_setting}/lora/"
    else:
        model_dir = f"model_disambiguation/{corpus_name}/{split_name}/{f_setting}/"

    parser = argparse.ArgumentParser()

    if f_setting == 'original_def_small_set':
        fine_tune_setting = 'original_title'
    elif f_setting == 'original_def':
        fine_tune_setting = 'original_title' 
    elif f_setting == 'ho_prime_others_not':
        fine_tune_setting = 'prime' 
    else:
        fine_tune_setting = f_setting

    parser.add_argument("--fine_tune_setting", default=fine_tune_setting)
    

    parser.add_argument("--lora", default=lora)

    # Rasel : kb
    parser.add_argument("--fine_tune", default=False)


    # parser.add_argument("--lego_test_data", default=f"{data_dir}{corpus_name}_test.json")
    parser.add_argument("--train_kb_prime", default=None)
    parser.add_argument("--use_title_during_testing", default=use_title_during_testing)
    # train_data_filename = f"{data_dir}{corpus_name}_{split_name}_ho_shuffled_candidates.json"
    # train_data_filename = f"{data_dir}{corpus_name}_{split_name}_ho.json"
    if split_name=='train':
        train_data_filename = f"{data_dir}train.json"
    parser.add_argument("--train_data", default=train_data_filename)

    parser.add_argument("--dev_data", default=f"{data_dir}test.json")
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=True)

    if before_fine_tune:
        parser.add_argument("--model",
                        default=f"model_disambiguation/zeshel_disambiguation_attention.pt")
        parser.add_argument("--eval_before_fine_tune", default=True)
        pred_data_file = f"{model_dir}pred_{split_name}_{fname}.json"
        parser.add_argument("--pred_data", default=pred_data_file)

    else:

        # parser.add_argument("--model",
        #                     default=f"model_disambiguation/{split_name}/zeshel_disambiguation_attention_{exp}_{corpus_name}.pt")


        model_path = f"{model_dir}zeshel_disambiguation_attention_{exp}_{corpus_name}_2.pt"
        parser.add_argument("--model",
                            default=model_path)
        parser.add_argument("--eval_before_fine_tune", default=False)
        parser.add_argument("--pred_data", default=f"{model_dir}{corpus_name}_pred_{exp}_{split_name}_{fname}.json")


    

    # rasel : load trained model
    parser.add_argument("--saved_pt_model",
                        default=None)

    parser.add_argument("--transformer_model",
                        default="roberta-base")

    # parser.add_argument("--transformer_model",
    #                     default="allenai/biomed_roberta_base")


    parser.add_argument("--cand_num", default=64,type=int)

    parser.add_argument("--type_loss", type=str,
                        default="sum_log_nce",
                        choices=["log_sum", "sum_log", "sum_log_nce",
                                    "max_min", "bce_loss"])
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--max_ent_len", default=256, type=int)
    parser.add_argument("--max_text_len", default=256, type=int)



    parser.add_argument("--train_kb", default=f"kb/{corpus_name}_kb.json")
    parser.add_argument("--dev_kb", default=f"kb/{corpus_name}_kb.json")

    parser.add_argument("--forgot_test_data", default=f"{data_dir}test.json")
    parser.add_argument("--lego_test_data", default=f"{data_dir}test.json")
    parser.add_argument("--star_test_data", default=f"{data_dir}test.json")
    parser.add_argument("--yugioh_test_data", default=f"{data_dir}test.json")

    parser.add_argument("--forgot_kb", default=f"kb/{corpus_name}_kb.json")
    parser.add_argument("--lego_kb", default=f"kb/{corpus_name}_kb.json")
    parser.add_argument("--star_kb", default=f"kb/{corpus_name}_kb.json")
    parser.add_argument("--yugioh_kb", default=f"kb/{corpus_name}_kb.json")

    parser.add_argument("--batch", default=4,type=int)

    # rasel : small
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--epochs", default=2)


    parser.add_argument("--warmup_proportion", default=0.1)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--simpleoptim", default=False)
    parser.add_argument("--clip", default=1)
    parser.add_argument("--info_token_num", default=3, type=int)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--logging_steps", default=100)
    parser.add_argument("--eval_step", default=10000, type=int)



    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    

    main(args)




