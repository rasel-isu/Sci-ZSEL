import argparse
import json
import os
from data_disambiguation import load_data, load_entities
from run_disambiguation_attention import main
# data = load_data('data/forgotten_realms.json')
# print(json.dumps(data[30], indent=2))

# data = load_data('data/dev_candidates.json')
# print(json.dumps(data[0], indent=2))

# data = load_entities('kb/train_kb.json')
# print(json.dumps(data[0], indent=2))


parser = argparse.ArgumentParser()

parser.add_argument("--model",
                    default="model_disambiguation/zeshel_disambiguation_attention.pt")
parser.add_argument("--transformer_model",
                    default="roberta-base")
# parser.add_argument("--transformer_model",
#                     default="allenai/biomed_roberta_base")

parser.add_argument("--type_loss", type=str,
                    default="sum_log_nce",
                    choices=["log_sum", "sum_log", "sum_log_nce",
                                "max_min", "bce_loss"])
parser.add_argument("--max_len", default=512, type=int)
parser.add_argument("--max_ent_len", default=256, type=int)
parser.add_argument("--max_text_len", default=256, type=int)

parser.add_argument("--train_data", default="data/train_candidates.json")
parser.add_argument("--dev_data", default="data/dev_candidates.json")
parser.add_argument("--train_kb", default="kb/train_kb.json")
parser.add_argument("--dev_kb", default="kb/val_kb.json")

parser.add_argument("--forgot_test_data", default="data/bc5cdr_test.json")
parser.add_argument("--lego_test_data", default="data/ncbi_test.json")
parser.add_argument("--star_test_data", default="data/medmentions_test.json")
parser.add_argument("--yugioh_test_data", default="data/cometa_test.json")

parser.add_argument("--forgot_kb", default="kb/mesh_kb.json")
parser.add_argument("--lego_kb", default="kb/mesh_kb.json")
parser.add_argument("--star_kb", default="kb/umls_kb.json")
parser.add_argument("--yugioh_kb", default="kb/snomedct_kb.json")


parser.add_argument("--batch", default=4,type=int)
parser.add_argument("--lr", default=4e-5, type=float)
parser.add_argument("--epochs", default=10)
parser.add_argument("--cand_num", default=10,type=int)
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
parser.add_argument("--do_train", action="store_true", default=False)
parser.add_argument("--do_eval", action="store_true", default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
main(args)




