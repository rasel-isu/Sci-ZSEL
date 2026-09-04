from datetime import datetime
import os
import time
from train_on_other_data import train_exp
from utils import Logger, print_run_config
from utils import CONFIG


# Everything below comes from config.json -> "res": { ... }.
# Valid "exp_list" entries (directories under datasets/<world>/res_format/<split>/):
#   "(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime"                          Sci-ZSEL w/o filter
#   "(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"          Sci-ZSEL
#   "synonym"                                                                  synonym baseline
#   "synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e"  Sci-ZSEL + Synonym
RES = CONFIG['res']

exp_settings = [{
    'corpus_name' : CONFIG['world'],
    'onto_name' : CONFIG['kb_name'],
    'split_name' : RES.get('split_name', 'train'),
    'use_title_during_testing' : RES.get('use_title_during_testing', True)
    }]

lora = RES.get('lora', False)

exp_list = RES.get('exp_list') or []
if not exp_list:
    raise ValueError('config.json: "res"."exp_list" is empty — nothing to train on')
exps = [{name: None} for name in exp_list]

os.makedirs("logs", exist_ok=True)
start_time = time.time()
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_path = os.path.join("logs",datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log"))
logger = Logger(log_path)
logger.log("============================================")
logger.log(f"Job started at: {start_timestamp}")
logger.log(f"Log file: {logger.log_path}")
logger.log("============================================")
print_run_config(logger, exp_settings, exps, lora)

for setting in exp_settings:
    for exp in exps:
        data_setting = list(exp.keys())[0]
        data_dir =  f"../{CONFIG['data_dir']}/res_format/{setting['split_name']}/{data_setting}/"
        train_exp(
            data_dir=data_dir,
            lora=lora,
            corpus_name = setting['corpus_name'],
            onto_name = setting['onto_name'],
            exp = exp,
            split_name = setting['split_name'],
            use_title_during_testing = setting['use_title_during_testing'])

end_time = time.time()
end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
elapsed_sec = int(end_time - start_time)
elapsed_hrs = (
    f"{elapsed_sec // 3600:02d}h "
    f"{(elapsed_sec % 3600) // 60:02d}m "
    f"{elapsed_sec % 60:02d}s"
)
logger.log("============================================")
logger.log(f"Job finished at:  {end_timestamp}")
logger.log(f"Started at:       {start_timestamp}")
logger.log(f"Elapsed time:     {elapsed_hrs} ({elapsed_sec}s)")
logger.log(f"Log file:         {logger.log_path}")
logger.log("============================================")
