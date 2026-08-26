from datetime import datetime
import os
import time

from train_on_other_data import train_exp
from eval_on_other_data import eval_exp
from data_preparation.utils import Logger, print_run_config
exp_settings = [
    {
        'corpus_name' : 'lpt',
        'onto_name' : 'lpt',
        'split_name' :'train',
        'use_title_during_testing' : True
    },
    {
        'corpus_name' : 'cmo',
        'onto_name' : 'cmo',
        'split_name' :'train',
        'use_title_during_testing' : True
    },
    {
        'corpus_name' : 'vt',
        'onto_name' : 'vt',
        'split_name' :'train',
        'use_title_during_testing' : True
    },
    {
        'corpus_name' : 'ncbi',
        'onto_name' : 'medic',
        'split_name' :'train',
        'use_title_during_testing' : True
    },
    {
        'corpus_name' : 'bc5cdr',
        'onto_name' : 'mesh',
        'split_name' :'train',
        'use_title_during_testing' : True
    }
]

# "(m1_e1)_(m3_e1)_(m4_e2)_and_their_plurals_34_rm_sm_e":None,
# "(m1_e1)_(m3_e1)":None,

lora = False
all_f_settings = [
    # 'prime',
    
    # {"(m1_e1)":'m1'},
    # {'(m4_e2)':'m4'},
    # {"(m1_e1)U(m3_e1)U(m4_e2)":None},

    # {"synonym":None},
    # {"synonymU(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e":None},
    # {"(m1_e1)U(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e":None},
    # {"(m1_e1)U(m3_e1)_multi_primeU(m4_e2)_multi_prime":None},
    
    {"synonymU(m3_e1)_multi_prime_rm_sm_eU(m4_e2)_multi_prime_rm_sm_e":None},
    {"synonymU(m1_e1)":None},
    

    # {"(m1_e1)_(m3_e1)_(m3_e1)_from_(m3s==e')_(m3==e's)":None},
    # {'(m4_e2)':'m4'},
    # {'(m5_e3)':'m5'},
    # {'(m6_e4)':'m6'},
    # {"(m1_e1)":None},
    # {"(m1_e1)_(m1s==e)_(m1==es)":'m1'},
    # {"(m4_e2)_(m4s==e')_(m4==e's)":'m4'},
    # {"(m4_e2)_rm_sm_e2":'m4'},
    # {"(m4_e2)_(m4s==e')_(m4==e's)_rm_sm_e2":'m4'},
    # {"(m5_e3)_(m5s==e')_(m5==e's)":'m5'},
    # {'(m3_e1)_(m4_e2)_(m5_e3)':None},
    # {'(m1_e1)_(m3_e1)_(m4_e2)_(m5_e3)':None},
    # {'(m6_e4)':'m6'},
    # {"(m6_e4)_(m6s==e')_(m6==e's)":'m6'},
    # {"(m4_e2)_(m4s==e')_(m4==e's)_rm_sm_e2":'m4'},
    # {"(m1_e1)_(m3_e1)_(m4_e2)":None},
    # {"(m1_e1)_(m3_e1)_(m4_e2)_34_rm_sm_e":None},
    # {"(m1_e1)_(m3_e1)_(m4_e2)_and_their_plurals":None},
    # {"(m1_e1)_(m3_e1)_(m4_e2)_and_their_plurals_34_rm_sm_e":None},
    # {"(m1_e1)_(m3_e1)_(m4_e2)_(m5_e3)_(m6_e4)":None},
    # {"(m1_e1)_(m3_e1)_(m4_e2)_(m5_e3)_(m6_e4)_3456_rm_sm_e":None}
    # 'original_def_small_set',
    # 'ho_prime_others_not',
    # {'original_def':None},
    # {"original_def_small_set":None},
    ]

os.makedirs("logs", exist_ok=True)
start_time = time.time()
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_path = os.path.join("logs",datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log"))
logger = Logger(log_path)
logger.log("============================================")
logger.log(f"Job started at: {start_timestamp}")
logger.log(f"Log file: {logger.log_path}")
logger.log("============================================")
print_run_config(logger, exp_settings, all_f_settings, lora)


for setting in exp_settings:
    for f_setting in all_f_settings:
        data_setting = list(f_setting.keys())[0]
        data_dir = f'data/blink/zshel_64_context_length/{setting["corpus_name"]}/{setting["split_name"]}/{data_setting}/'
        train_exp(
            data_dir=data_dir,
            lora=lora,
            corpus_name = setting['corpus_name'],
            onto_name = setting['onto_name'],
            exp = f_setting,
            split_name = setting['split_name'],
            use_title_during_testing = setting['use_title_during_testing'],
            # both_set=True
        )

# setting = {
#         'onto_name' : 'bc5cdr',
#         'split_name' :'train',
#         'use_title_during_testing' : True
#     }
# data_dir = f'data/blink/zshel_64_context_length/{setting["onto_name"]}/{setting["split_name"]}/before_fine_tune/'
# eval_exp(
#     data_dir=data_dir,
#     f_setting = 'before_fine_tune',
#     lora=lora,
#     corpus_name = setting['onto_name'],
#     exp = 'before_fine_tune',
#     split_name = setting['split_name'],
#     use_title_during_testing = True,
#     before_fine_tune=True)


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
