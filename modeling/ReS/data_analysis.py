import json
import math
import re
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import pandas as pd
import numpy as np
from eval import cat_eval, eval_res_dataset
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from data_preparation.utils import summary_for_test_ent_appears_in_train


def count_mc(corpusfile, kbfile):
    with open(corpusfile) as f:
        corpus = json.load(f)
    with open(kbfile) as f:
        kb = json.load(f)
    mc_count = 0
    ho_count = 0
    ho_matched_count = 0
    for i in corpus:
        text = i['text'][0]
        m = re.search(r'\[E1\](.*?)\[\\E1\]', text).group(1).strip()
        t = kb[i['mention_data']['kb_id'][0]]['title'].strip() 
        if m.lower() == t.lower():
            ho_count+=1
            if i['linked']==1.0:
                ho_matched_count+=1
        
        if "(" in t:
            print(t)
            print(m)
            print('_'*10)
            
            td = t.split('(')[0].strip()
            if td.lower() == m.lower():
                mc_count+=1
    print(f'{corpusfile}')
    print(f'Matched MC : {mc_count}')
    print(f'HO : {ho_matched_count}/{ho_count}={ho_matched_count/ho_count}')
    cat_eval(0, corpusfile, kbpath=kbfile)
    print('*'*100)


# def process_men_eq_onto_ent_and_men_not_eq_gt():

def count_men_eq_onto_ent_and_men_not_eq_gt(corpusfile, kbfile):
    with open(corpusfile) as f:
        corpus = json.load(f)
    with open(kbfile) as f:
        kb = json.load(f)

    c_name = corpusfile.split('/')[1].split('_')[0]
    print(c_name)

    # tind_kb = {}
    # for i in kb:
    #     if kb[i]['title'].lower() in tind_kb:
    #         print(i)
    #         print(tind_kb[kb[i]['title'].lower()])
    #         print('_'*10)

    #     tind_kb[kb[i]['title'].lower()] = i
    # if len(kb)!=len(tind_kb):
    #     print(f'KB items: {len(kb)}, converted KB items : {len(tind_kb)} \nthis is not same!')
    #     return False

    men_eq_onto_ent_count = 0
    men_not_eq_gt_count = 0
    count_gt_was_same = 0
    count_data = []

    for c in tqdm(corpus):
        text = c['text']
        m = text.split('[E1]')[1].split('[\\E1]')[0].strip()
        for i in kb:
            onto_title = kb[i]['title']
            if m.lower() == onto_title.lower():
                men_eq_onto_ent_count+=1
                gtid = c['mention_data']['kb_id']
                gttitle = kb[gtid]['title']
                if m.lower() != gttitle.lower():
                    men_not_eq_gt_count+=1
                    count_data.append({
                        'mention___': m, 
                        'onto_title':onto_title,
                        'gt_title__':gttitle,
                        'onto_id___':i,
                        'gt_id_____':gtid
                        })

    with open(f'analysis/men_eq_onto_ent_and_men_not_eq_gt_{c_name}.json', 'w') as f:
        json.dump({**{'men_eq_onto_ent_count':men_eq_onto_ent_count,
                      'men_not_eq_gt_count':men_not_eq_gt_count
                      }, **{'data':count_data}}, f, indent=1)

    print(f'men_eq_onto_ent_count : {men_eq_onto_ent_count} \n& men_not_eq_gt_count {men_not_eq_gt_count}')

def get_diff_btwn_gt_and_pred_grag(pred_file, kbpath, mention_token=['[E1]', '[\\E1]']):
    with open(pred_file) as f:
        corpus = json.load(f)

    source_key="pred_id___"
    target_key="gt_id_____"

    pred_and_gt = []
    for c in tqdm(corpus):
        text = c['mention_context']
        m = text.split(mention_token[0])[1].split(mention_token[1])[0].strip()
        frist_candidate = c['retrieved_candidates'][0]
        pred_and_gt.append({
                        'mention___': m, 
                        'pred_title':frist_candidate['title'],
                        'gt_title__':c['ground_truth']['title'],
                        source_key:frist_candidate['id'],
                        target_key:c['ground_truth']['id'],
                        'mention_context':c['mention_context']
                        })
    gt_vs_prd = pred_file.replace('.json', '_gt_vs_prd.json')
    with open(gt_vs_prd, 'w') as f:
        json.dump(pred_and_gt, f, indent=1)

    connection_between_gtt_and_oet(gt_vs_prd, kbpath,'mesh', source_key, target_key)



def get_mesh_relations():
    G = nx.MultiDiGraph()
    rfile = [
        'relations_desc2025.json',
        'relations_pa2025.json',
        'relations_qual2025.json',
        'relations_supp2025.json'
        ]
    for i in rfile:
        with open('/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/onto/mesh/'+i) as f:
            data = json.load(f)
        for source, relations in data.items():
            for rel_type, targets in relations.items():
                for target in targets:
                    G.add_edge(source, target, relation=rel_type)
        print(f"Total nodes: {G.number_of_nodes()}")
        print(f"Total edges: {G.number_of_edges()}")

    return G


def get_umls_relations():
    G = nx.MultiDiGraph()
    rfile = ['umls_rel.json']
    for i in rfile:
        with open('/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/MedMentions/full/data/'+i) as f:
            data = json.load(f)
        for source, relations in data.items():
            for rel_type, targets in relations.items():
                for target in targets:
                    G.add_edge(source, target, relation=rel_type)
        print(f"Total nodes: {G.number_of_nodes()}")
        print(f"Total edges: {G.number_of_edges()}")

    return G
def get_snomedct_relations():
    G = nx.MultiDiGraph()
    rfile = ['snomedct_rels.json', 'snomedct_rels_stated.json']
    for i in rfile:
        with open('/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/cometa/'+i) as f:
            data = json.load(f)
        for source, relations in data.items():
            for rel_type, targets in relations.items():
                for target in targets:
                    G.add_edge(source, target, relation=rel_type)
        print(f"Total nodes: {G.number_of_nodes()}")
        print(f"Total edges: {G.number_of_edges()}")

    return G
def check_onto_gt_matching(onto_id, gt_id, relations):
    if onto_id in relations:
        ent_relations = relations[onto_id]
        for rel in ent_relations:
            if gt_id in ent_relations[rel]:
                return (onto_id, rel, gt_id)
        return ent_relations
    
def path_with_relations(G, source, target):

    try:
        if not nx.has_path(G, source, target):
            return [], '', -1
    except Exception as e:
        return [], '', -1

    route = ''
    depth = []
    path = nx.shortest_path(G, source=source, target=target)
    for d, (u, v) in enumerate(zip(path[:-1], path[1:]), 1):
        depth.append(d)
        edge_data = G.get_edge_data(u, v)
        edge = edge_data[0]
        route += f"{u}, {edge['relation']}, {v} | "
    hops = len(path)-1
    # if hops>1:
    #     print(hops)

    return path, route.strip(), hops

def connection_between_gtt_and_oet(filepath, kbpath, onto='mesh', source_key="onto_id___", target_key="gt_id_____"):
    if onto=='mesh':
        G = get_mesh_relations()
    elif onto=='umls':
        G = get_umls_relations()
    elif onto=='snomedct':
        G = get_snomedct_relations()
    

    with open(filepath) as f:
        mentions = json.load(f)
    with open(kbpath) as f:
        kb = json.load(f)

    conns = []
    no_conn = 0
    for m in tqdm(mentions):
        source_node = m[source_key]
        target_node = m[target_key]

        # if not (source_node == 'D000083242' and target_node == 'D002544'):
        #     continue

        path_id, route, hops = path_with_relations(G, source_node, target_node)
        if hops==-1:
            no_conn+=1

        route_title=route
        for pid in path_id:
            route_title=route_title.replace(pid, kb[pid]['title'])

        
        conns.append({
                source_key : source_node, 
                target_key :target_node,
                'route_as_id':route,
                'route_as_title':route_title,
                'hops':hops,
                'mention' : m['mention___'],
                'mention_context': m['mention_context']
            })
    conns = sorted(conns, key=lambda x: x['hops'], reverse=True)

    outfile = filepath.replace('.json', '_connection_gtt_oet.json')

    hop_info = get_hop_count(conns, outfile)

    with open(outfile, 'w') as f:
        json.dump({'hop_stats':hop_info, 'hops':conns}, f, indent=1)

    print(f'Total mentions: {len(mentions)}\nnot connected : {no_conn}\n{no_conn}')
def boxplot_custom_percentiles_v2(data, percentiles=[10, 30, 50, 70, 90]):
    fig, ax = plt.subplots(figsize=(5, 5))
    perc_values = np.percentile(data, percentiles)
    p_low, p_q1, p_median, p_q3, p_high = perc_values
    box_width = 0.3
    x_center = 1
    box_height = p_q3 - p_q1
    box = Rectangle((x_center - box_width/2, p_q1), box_width, box_height,
                   facecolor="#0A5693", edgecolor='black', alpha=0.7)
    ax.add_patch(box)
    
    # median line
    # ax.plot([x_center - box_width/2, x_center + box_width/2], 
    #         [p_median, p_median], 'r-', linewidth=2, label='Median', c='red')
    
    # whiskers
    ax.plot([x_center, x_center], [p_q1, p_low], 'k-', linewidth=1)  # Lower whisker
    ax.plot([x_center, x_center], [p_q3, p_high], 'k-', linewidth=1)  # Upper whisker
    
    # caps
    cap_width = box_width * 0.5
    ax.plot([x_center - cap_width/2, x_center + cap_width/2], 
            [p_low, p_low], 'k-', linewidth=1)  # Lower cap
    ax.plot([x_center - cap_width/2, x_center + cap_width/2], 
            [p_high, p_high], 'k-', linewidth=1)  # Upper cap
    
    # Optional: Add outliers (values outside custom percentiles)
    outliers_low = data[data < p_low]
    outliers_high = data[data > p_high]
    
    if len(outliers_low) > 0:
        ax.scatter([x_center] * len(outliers_low), outliers_low, 
                  c='black', s=20, alpha=0.6, label='Outliers')
    if len(outliers_high) > 0:
        ax.scatter([x_center] * len(outliers_high), outliers_high, 
                  c='black', s=20, alpha=0.6)
    
    # Formatting
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([x_center])
    ax.set_yticks(range(np.min(data), np.max(data)+1, 1))
    ax.set_xticklabels(['Percentiles'])
    ax.set_title(f'Percentiles are : {percentiles}%')
    ax.set_ylabel('Number of hops')
    ax.grid(True, alpha=0.8)
    ax.legend()
    
    # Add percentile labels
    custom_labels = {-1.0: "  No connection"}
    for i, (perc, val) in enumerate(zip(percentiles, perc_values)):
        if custom_labels and val in custom_labels:
            label_text = f'{perc}%'
        else:
            label_text = f'{perc}%'
        
        ax.text(1.4, val, label_text, 
                verticalalignment='center', fontsize=10)
    
    plt.tight_layout()
    return fig, ax

def get_hop_count(data, json_file):
    df = pd.DataFrame(data)
    counts = df['hops'].value_counts().to_dict()
    samples = []
    samples_hist = []
    for hop, count in counts.items():
        samples_hist.extend([int(hop)] * count)
        # if hop == -1:
        #     continue
        samples.extend([int(hop)] * count)

    samples_array = np.array(samples)
    hops_df = pd.DataFrame(samples_array, columns=["hops"])
    percentiles = hops_df["hops"].quantile([0.70, 0.80, 0.90]).to_dict()
    counts = dict(sorted(counts.items()))
    total = sum(counts.values())
    counts_prct = {}
    xtik_hop_label = {}
    for item in counts:
        perct = round(counts[item]/total, 2)
        pc_r = int(round(perct*100, 0))
        counts_prct[f'{item} ({perct})'] = counts[item]
        if item == -1:
            xtik_hop_label[item] = f'({counts[item]}, {pc_r}%) Not'
        else:
            xtik_hop_label[item] = f'({counts[item]}, {pc_r}%) {item}'
        

    


    mpl.rcParams['font.size'] = 25
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'

    xtik_hops = range(-1, 11, 1)
    plt.figure(figsize=(10, 25))
    plt.hist(samples_hist, bins=xtik_hops, edgecolor='black', align='left', rwidth=0.8)
    # plt.title("Hop Count Distribution")
    plt.xlabel("(N. of sample,%) Hop")
    plt.ylabel("Hop frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(xtik_hops, [xtik_hop_label[hop] if hop in xtik_hop_label else str(hop) for hop in xtik_hops], rotation=75)
    hist_file = json_file.replace('.json', '_hist.png')
    plt.savefig(hist_file, dpi=300)
    plt.close()  

 
    mpl.rcParams['font.size'] = 10
    filtered_hops = hops_df[hops_df["hops"] != -1]["hops"]
    # plt.figure(figsize=(4, 4))
    ax =sns.boxplot(filtered_hops)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(range(int(ymin), int(ymax) + 1))
    # ax.set_yticks(yticks)
    # fig2, ax2 = boxplot_custom_percentiles_v2(hops_df['hops'].values, [1, 25, 50, 75, 100])
    box_file = json_file.replace('.json', '_box.png')
    plt.savefig(box_file, dpi=300)
    plt.close()    

    values = list(counts.keys())
    if -1 in values:
        values.remove(-1)
    highest_hops = max(values)
    avg_hops = sum(values) / len(values)
    return {
        'total_sample':total,
        'highest_hops':highest_hops,
                    'avg_hops':avg_hops,
                    'percentiles':percentiles,
                    'hop_counts':counts_prct
                    }

def plot_acc(dir, name, x,y):
    plt.plot(x, y, marker='o')
    plt.title("Accuracy and Epoch on Val set")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{dir}/acc_epoch_{name}.png", dpi=300)
    plt.close()



def count_ent(filepath):
    with open(f'{filepath}') as f:
        ent = json.load(f)
    print(f'{len(ent)} : {filepath}')

def compare_blink_and_res(bpath, rpath):
    with open(f'{bpath}') as f:
        bdata = json.load(f)['63']['HO']
    with open(f'{rpath}') as f:
        rdata = json.load(f)['63']['HO']
    
    print(f'length')
    print(f'{bdata["matched"]} : {bpath}')
    print(f'{rdata["matched"]} : {rpath}')
    
    i_bdata=[i for i in bdata['items'] if i['matched']]
    i_rdata=[i for i in rdata['items'] if i['matched']]
    for i in i_bdata:
        m = i['mention']
        match = False
        for j in i_rdata:
            if j['mention']==m:
                match = True
        if not match:
            print(0)
        

    print(0)

def compare_prime_hist(hist_file, train_file, epoch):
    with open(f'{hist_file}') as f:
        prime_ent = json.load(f)
    print(f'{len(prime_ent)} : {hist_file}')
    with open(f'{train_file}') as f:
        gt_ent = json.load(f)
    print(f'{len(gt_ent)} : {train_file}')

    count = {}
    for i in gt_ent:
        gtid = i['mention_data']['kb_id']
        if gtid in count:
            count[gtid]+=1
        else:
            count[gtid]=1

    all_matched = True
    mismatched = {}
    for g in count:
        c_gt =  count[g]
        c_pu = prime_ent[g]['count']
        if not c_gt*epoch == c_pu:
            all_matched = False
            mismatched[g] = prime_ent[g]
            mismatched[g]['train_gt_count'] = c_gt

    if not all_matched:
        print(json.dumps(mismatched, indent=1))
        print('The above primes count didnt match!')
    else:
        print('prime usage history and train gt matched, so it looks ok!')

    # print(json.dumps(count, indent=1))

def compare_all_settings_hist(onto, files, save_to_dir):
    data = {}
    for file in files:
        with open(f'{files[file]}') as f:
            conns = json.load(f)['hops']
            df = pd.DataFrame(conns)
            counts = df['hops'].value_counts().to_dict()
            data[file] = counts

    # Prepare layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Set common x-ticks and y-limit
    all_keys = sorted(set(k for d in data.values() for k in d))
    x_ticks = list(all_keys)
    y_max = max(max(d.values()) for d in data.values()) + 50

    # Plot each histogram
    for i, (title, count_dict) in enumerate(data.items()):
        ax = axes[i]
        total = sum(count_dict.values())
        counts = [count_dict.get(k, 0) for k in x_ticks]
        bars = ax.bar(x_ticks, counts, color="#0A5693", edgecolor='black')

        # Annotate bars with count and percentage
        for bar, c in zip(bars, counts):
            pct = (c / total) * 100
            pct = int(round(pct, 0))

            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f'{c}\n{pct}%', ha='center', va='bottom', fontsize=9)

        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Hops')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(f"{save_to_dir}{onto}_subplots.png", dpi=500)

    methods = list(data.keys())
    all_labels = sorted(set(k for d in data.values() for k in d))  # all possible class labels
    label_strs = [str(k) for k in all_labels]

    x = np.arange(len(all_labels))  # positions for groups
    width = 0.2  # width of each bar
    # Define color palette
    colors = ['blue', 'orange', 'green', 'red']
    # Prepare figure
    fig, ax = plt.subplots(figsize=(18, 10))
    # Plot each method as separate bar set
    for i, (method, color) in enumerate(zip(methods, colors)):
        counts = [data[method].get(k, 0) for k in all_labels]
        total = sum(counts)
        positions = x + (i - 1.5) * width
        bars = ax.bar(positions, counts, width, label=method, color=color)

        # Add count and % on top of each bar
        for bar, c in zip(bars, counts):
            pct = (c / total) * 100 if total > 0 else 0
            pct = int(round(pct, 0))
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'      {c}, {pct}%', ha='center', va='bottom', fontsize=12, rotation=90)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(label_strs)
    ax.set_xlabel('Hops')
    ax.set_ylabel('No. of sample')
    ax.set_title('Hop distribution across methods')
    ax.legend()
    ax.set_ylim(0, max(max(data[m].get(k, 0) for k in all_labels) for m in methods) + 50)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_to_dir}{onto}_single_hist.png", dpi=500)
                
def categorywise_prime_count(source_file, prime_file):
    with open(f'{source_file}') as f:
        data = json.load(f)
    with open(f'{prime_file}') as f:
        prime = json.load(f)
    cat_wise_gt_matched = {
            "HO":{'count':0,'has_prime':0, 'unigt_id':{}}, 
            "MINT":{'count':0,'has_prime':0, 'unigt_id':{}}, 
            "LO":{'count':0,'has_prime':0, 'unigt_id':{}},
            "NO":{'count':0,'has_prime':0, 'unigt_id':{}}}

    for i in data:
        mention = i['mention']
        title = i['ground_truth']['title']
        gtid = i['ground_truth']['id']
        mention_lower = mention.lower()
        title_lower = title.lower()
        if mention_lower == title_lower:
            cat_wise_gt_matched["HO"]['count'] += 1
            cat_wise_gt_matched["HO"]['unigt_id'][gtid]=title
            if gtid in prime:
                cat_wise_gt_matched["HO"]['has_prime'] += 1


        elif mention_lower in title_lower and title_lower != mention_lower:
            cat_wise_gt_matched["MINT"]['count'] += 1
            cat_wise_gt_matched["MINT"]['unigt_id'][gtid]=title
            

        else:
            lo = False
            mention_words = mention_lower.split()
            for word in mention_words:
                if word in title_lower:
                    lo = True
                    break
            if lo:
                cat_wise_gt_matched["LO"]['count'] += 1
                cat_wise_gt_matched["LO"]['unigt_id'][gtid]=title
            else:
                cat_wise_gt_matched["NO"]['count'] += 1
                cat_wise_gt_matched["NO"]['unigt_id'][gtid]=title

    return cat_wise_gt_matched
            


source_file = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/ncbi/train_grag.json'
prime_file = 'kb/ncbi_prime_train_defi.json'
categorywise_prime_count(source_file, prime_file)




# prompt_el_file = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/ncbi-disease/test.json'
# check_correctness_of_prepared_data('data/blink/zshel/ncbi_test_ho.json', prompt_el_file)
# check_correctness_of_prepared_data('data/blink/zshel/ncbi_test.json', prompt_el_file)
    
# prompt_el_file = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/bc5cdr/test.json'
# check_correctness_of_prepared_data('data/blink/zshel/bc5cdr_test_ho.json', prompt_el_file)
# check_correctness_of_prepared_data('data/blink/zshel/bc5cdr_test.json', prompt_el_file)




# compare_prime_hist(
#     hist_file='model_disambiguation/test/prime/ncbi_prime_history.json', 
#     train_file='data/blink/zshel/ncbi_test_ho.json', 
#     epoch=1)

# compare_prime_hist(
#     hist_file='model_disambiguation/test/prime/prime_history.json', 
#     train_file='data/blink/zshel/bc5cdr_test_ho.json', 
#     epoch=8)


# count_ent('kb/ncbi_prime_train.json')
# count_ent('kb/ncbi_prime_test.json')
# count_ent(f"data/ncbi_train_ho.json")
# count_ent(f"/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/ncbi-disease/train_grag_generated_ho.json")
# count_ent(f"/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/ncbi-disease/train_grag.json")
# count_ent(f"data/ncbi_test.json")
# count_ent(f"/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/ncbi-disease/test_grag.json")

# count_ent(f"data/ncbi_test_ho.json")
# count_ent(f"/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/ncbi-disease/test_grag_generated_ho.json")
# count_ent(f"/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/datasets/ncbi-disease/test_grag.json")
# summary_for_test_ent_appears_in_train(train_file, test_file)



# count_ent(f'kb/bc5cdr_prime_test.json')
# count_ent(f'kb/bc5cdr_prime_train.json')

# acc_lis = [12, 13, 58, 67, 69]
# plot_acc('ncbi', [i+1 for i in range(len(acc_lis))], acc_lis)
# acc_lis = [4, 46, 70, 73, 77]
# plot_acc('bc5cdr', [i+1 for i in range(len(acc_lis))], acc_lis)
        
# # For ReS
# split_name = 'test'
# exp = 'only_definition'
# onto = 'ncbi'
# predfile = f'model_disambiguation/{split_name}/{exp}/{onto}_pred_{exp}_test_with_test_title_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')
# onto = 'bc5cdr'
# predfile = f'model_disambiguation/{split_name}/{exp}/{onto}_pred_{exp}_test_with_test_title_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')

# exp = 'prime'
# onto = 'ncbi'
# predfile = f'model_disambiguation/{split_name}/{exp}/{onto}_pred_{exp}_test_with_test_title_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')
# onto = 'bc5cdr'
# predfile = f'model_disambiguation/{split_name}/{exp}_/{onto}_pred_{exp}_test_with_test_title_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')

# onto = 'ncbi'
# predfile = f'model_disambiguation/{split_name}/{exp}/{onto}_pred_{exp}_test_with_test_title_grag_before_fine_tune.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')
# onto = 'bc5cdr'
# predfile = f'model_disambiguation/{split_name}/{exp}/{onto}_pred_{exp}_test_with_test_title_grag_before_fine_tune.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')

# exp = 'original_title'
# onto = 'ncbi'
# predfile = f'model_disambiguation/{split_name}/{exp}/{onto}_pred_{exp}_test_with_test_title_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')
# onto = 'bc5cdr'
# predfile = f'model_disambiguation/{split_name}/{exp}/{onto}_pred_{exp}_test_with_test_title_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json')


# onto = 'ncbi'
# splitname = 'test'
# dirname = f'model_disambiguation/{splitname}/'

# files = {
#     'without fine-tuning':f'{dirname}prime/{onto}_pred_prime_test_with_test_title_grag_before_fine_tune_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with original+def':f'{dirname}original_title/{onto}_pred_original_title_test_with_test_title_grag_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with prime':f'{dirname}prime/{onto}_pred_prime_test_with_test_title_grag_gt_vs_prd_connection_gtt_oet.json',
#     'only definition':f'{dirname}only_definition/{onto}_pred_only_definition_test_with_test_title_grag_gt_vs_prd_connection_gtt_oet.json'
# }
# compare_all_settings_hist(onto, files, dirname)

# onto = 'bc5cdr'
# splitname = 'test'
# dirname = f'model_disambiguation/{splitname}/'

# files = {
#     'without fine-tuning':f'{dirname}prime/{onto}_pred_prime_test_with_test_title_grag_before_fine_tune_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with original+def':f'{dirname}original_title/{onto}_pred_original_title_test_with_test_title_grag_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with prime':f'{dirname}prime/{onto}_pred_prime_test_with_test_title_grag_gt_vs_prd_connection_gtt_oet.json',
#     'only definition':f'{dirname}only_definition/{onto}_pred_only_definition_test_with_test_title_grag_gt_vs_prd_connection_gtt_oet.json'
# }
# compare_all_settings_hist(onto, files, dirname)


# # # For BLINK
# onto = 'ncbi'
# predfile = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/crossencoder_before_fine_tune_crossenc/crossencoder_predictions_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json', mention_token=['[MENTION_START]', '[MENTION_END]'])
# predfile = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/crossencoder_after_prime_fine_tune_crossenc/crossencoder_predictions_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json', mention_token=['[MENTION_START]', '[MENTION_END]'])
# predfile = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/crossencoder_after_original_def_fine_tune_crossenc/crossencoder_predictions_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json', mention_token=['[MENTION_START]', '[MENTION_END]'])

# onto = 'bc5cdr'
# predfile = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/crossencoder_before_fine_tune_crossenc/crossencoder_predictions_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json', mention_token=['[MENTION_START]', '[MENTION_END]'])
# predfile = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/crossencoder_after_prime_fine_tune_crossenc/crossencoder_predictions_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json', mention_token=['[MENTION_START]', '[MENTION_END]'])
# predfile = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/crossencoder_after_original_def_fine_tune_crossenc/crossencoder_predictions_grag.json'
# get_diff_btwn_gt_and_pred_grag(predfile, f'kb/{onto}_kb.json', mention_token=['[MENTION_START]', '[MENTION_END]'])

# onto = 'ncbi'
# splitname = 'test'
# dirname = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/'

# files = {
#     'without fine-tuning':f'{dirname}crossencoder_before_fine_tune_crossenc/{splitname}/crossencoder_predictions_grag_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with original+def':f'{dirname}crossencoder_after_original_def_fine_tune_crossenc/{splitname}/crossencoder_predictions_grag_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with prime':f'{dirname}crossencoder_after_prime_fine_tune_crossenc/{splitname}/crossencoder_predictions_grag_gt_vs_prd_connection_gtt_oet.json'
# }
# compare_all_settings_hist(onto, files, dirname)

# onto = 'bc5cdr'
# splitname = 'test'
# dirname = f'/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/models/{onto}/'
# files = {
#     'without fine-tuning':f'{dirname}crossencoder_before_fine_tune_crossenc/{splitname}/crossencoder_predictions_grag_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with original+def':f'{dirname}crossencoder_after_original_def_fine_tune_crossenc/{splitname}/crossencoder_predictions_grag_gt_vs_prd_connection_gtt_oet.json',
#     'fine-tuned with prime':f'{dirname}crossencoder_after_prime_fine_tune_crossenc/{splitname}/crossencoder_predictions_grag_gt_vs_prd_connection_gtt_oet.json'
# }
# compare_all_settings_hist(onto, files, dirname)


# get_hop_count()
# connection_between_gtt_and_oet('analysis/men_eq_onto_ent_and_men_not_eq_gt_bc5cdr.json')
# connection_between_gtt_and_oet('analysis/men_eq_onto_ent_and_men_not_eq_gt_ncbi.json') 
# connection_between_gtt_and_oet('analysis/men_eq_onto_ent_and_men_not_eq_gt_medmentions.json', 'umls') 
# connection_between_gtt_and_oet('analysis/men_eq_onto_ent_and_men_not_eq_gt_cometa.json', 'snomedct') 



# count_men_eq_onto_ent_and_men_not_eq_gt('data/ncbi_test.json', 'kb/mesh_kb.json')
# count_men_eq_onto_ent_and_men_not_eq_gt('data/bc5cdr_test.json', 'kb/mesh_kb.json')
# count_men_eq_onto_ent_and_men_not_eq_gt('data/cometa_test.json', 'kb/snomedct_kb.json')
# count_men_eq_onto_ent_and_men_not_eq_gt('data/medmentions_test.json', 'kb/umls_kb.json')

# count_mc('pred/bc5cdr_pred.json', "kb/mesh_kb.json")
# count_mc('pred/ncbi_pred.json', "kb/mesh_kb.json")
# count_mc('pred/medmentions_pred.json',"kb/umls_kb.json")

# data = 'bm25'
# mcfile = f'data/{data}/ncbi_test_ho.json'
# kbpath = 'kb/mesh_kb.json'
# eval_res_dataset(mcfile, kbpath)
# mcfile =  f'data/{data}/ncbi_test.json'
# eval_res_dataset(mcfile, kbpath)


# data = 'data/blink/zshel/'
# mcfile = f'{data}bc5cdr_test_ho.json'
# kbpath = 'kb/mesh_kb.json'
# eval_res_dataset(mcfile, kbpath)
# mcfile =  f'{data}bc5cdr_test.json'
# eval_res_dataset(mcfile, kbpath)


# data = 'data/blink/zshel/'
# mcfile = f'{data}ncbi_test_ho.json'
# kbpath = 'kb/mesh_kb.json'
# eval_res_dataset(mcfile, kbpath)
# mcfile =  f'{data}ncbi_test.json'
# eval_res_dataset(mcfile, kbpath)
# mcfile =  f'{data}ncbi_test_ho_shuffled_candidates.json'
# eval_res_dataset(mcfile, kbpath)
# kbpath = 'kb/bc5cdr_kb.json'
# mcfile =  f'{data}bc5cdr_test_ho_shuffled_candidates.json'
# eval_res_dataset(mcfile, kbpath)




# compare_blink_and_res('/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/BLINK/output/bc5cdr/zshel_tarined/predictions_category_info.json',
#                        '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/Read-and-Select/data/blink/zshel/bc5cdr_test_category_info.json')



