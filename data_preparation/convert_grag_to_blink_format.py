from utils import CONFIG, grag_to_blink_original_def

def make_data_for_testset(kb_file_path, world, split_name, source_file):
    print(f"{'_'*20}{world}{'_'*20}")
    out_path = f'{CONFIG["data_dir"]}/blink_format/train/original_data/'
    grag_to_blink_original_def(source_dir=CONFIG["data_dir"],source_file=source_file, 
            kb_file_path=kb_file_path,title_key='name',defi_key='def',world=world, 
            split_name=split_name,out_file='test.jsonl',out_path=out_path, skip_sample_if_ent_not_in_kb=True,
            only_ho=False, ho_prime_others_not=False)

def make_data_for_trainset(kb_file_path, world, split_name, source_file):
    print(f"{'_'*20}{world}{'_'*20}")
    out_path = f'{CONFIG["data_dir"]}/blink_format/train/original_data/'
    grag_to_blink_original_def(source_dir=CONFIG["data_dir"], source_file=source_file, 
            kb_file_path=kb_file_path,title_key='name',defi_key='def',world=world, 
            split_name=split_name,out_file='train.jsonl',out_path=out_path,
            skip_sample_if_ent_not_in_kb=True, # skip if gt ent not in kb
            only_ho=False, ho_prime_others_not=False,
            has_gt=CONFIG["has_ground_truth"] # dont evaluate if no GT
            )

def main():
    world = CONFIG['world']
    kb_file = CONFIG['kb_file']
    data_dir = CONFIG['data_dir']
    kb_file_path = f'{data_dir}{kb_file}'
    
    ## Convert GRAG to BLINK

    # for trainset
    make_data_for_trainset(kb_file_path, world, 'train', f'train_grag.json')

    # for testset
    make_data_for_testset(kb_file_path, world, 'test', f'test_grag.json')

if __name__ == "__main__":
    main()
