import json
import time
from tqdm import tqdm
from langchain_core.messages import SystemMessage, HumanMessage
from utils import CONFIG
import torch
import ollama
from langchain_ollama import ChatOllama

class LLMSelector():


    def __init__(self):
        self.seed = 0
        torch.cuda.manual_seed_all(self.seed)
        self.temperature = 0.0

    def get_ollama(self, base_model, num_ctx=25000):
        base_url = "http://127.0.0.1:11435"
        llm = ChatOllama(model= base_model, 
                          base_url=base_url,
                          request_timeout=120*60*60, temperature=self.temperature,
                          num_ctx = num_ctx)

        self.ollama_client = ollama.Client(host=base_url)
        return llm
    


class DataGen():
    def __init__(self, corpus_name, model_name='meta-llama/Llama-3.2-1B-Instruct', onto_filepath=''):
        self.model_name = model_name
        self.corpus_name = corpus_name
        self.token_stats = []
        self.tot_sys = self.tot_hum = self.tot_all= self.tot_out = 0
        self.onto_filepath = onto_filepath
        self.kb = self.load_kb()

    def load_kb(self):
        with open(self.onto_filepath) as f:
            kb = json.load(f)
        return kb

    def set_llm(self):
        llm_selector = LLMSelector()
        self.llm  = llm_selector.get_ollama(self.model_name, num_ctx=5000)
        self.ollama_client = llm_selector.ollama_client

    def count_tokens(self, text):
        resp = self.ollama_client.generate(
            model=self.model_name,
            prompt=text,
            raw=True,                       # don't apply chat template, count raw text
            options={'num_predict': 0},     # evaluate prompt only, generate nothing
        )
        return resp.get('prompt_eval_count', 0)
        
    def generate_ent_name(self, item):
        prompt, sys_tokens, hum_tokens = self.prompt_only_def_multiple_prime_from_entity(item)

        result  = self.llm.invoke(prompt)
        response = result.content

        output_tokens = self.count_tokens(response)
        total_tokens = sys_tokens+hum_tokens
        self.token_stats.append({
                "id":item['id'],
                "system_tokens": sys_tokens,
                "human_tokens": hum_tokens,
                "total_tokens": total_tokens,
                "output_tokens": output_tokens,
            })

        self.tot_sys += sys_tokens
        self.tot_hum += hum_tokens
        self.tot_all += total_tokens
        self.tot_out += output_tokens

        return response

    
    def prompt_only_def_multiple_prime_from_entity(self, item):
        with open(f'prompts/{self.corpus_name}/system.txt', "r") as file:
            system = file.read()
        system = system.replace("{", '{{').replace("}", '}}')
        with open(f'prompts/{self.corpus_name}/human.txt', "r") as file:
            question = file.read()
        defi = item['def']
        question = question.replace("{definition}", defi)
        question = question.replace("{", '{{').replace("}", '}}')

        sys_tokens = self.count_tokens(system)
        hum_tokens = self.count_tokens(question)

        prompt = [SystemMessage(content=system)]+[HumanMessage(content=question)]
        return prompt, sys_tokens, hum_tokens

    
def generate_alias_from_entity(corpus_name, onto_name, source_ent_filepath, onto_filepath,gen_key='newly_generated_name'):
    with open(source_ent_filepath) as f:
        data = json.load(f)
       
    gen = DataGen(
        corpus_name,model_name="llama3.2:3b-instruct-fp16", 
        onto_filepath=onto_filepath)
    
    gen.set_llm()

    c = 0
    er_c = 0
    generated_prime = []
    start = time.time()
    for i in tqdm(data):
        try:
            item = data[i]
            newly_generated_name = gen.generate_ent_name(item)
            item[gen_key] = newly_generated_name
            generated_prime.append({
                "document_id": item['id'],
                    "def": item['def'],
                    "old_name": item['name'],
                    "newly_generated_name": newly_generated_name
            })
            c+=1
        except Exception as e:
            print(f'Error :  {e}')
            er_c+=1

    end = time.time()
    taken = end - start
    human_readable = time.strftime("%H:%M:%S", time.gmtime(taken))
    print(f"Execution time: {human_readable}")

    print(f"Error : {er_c}")
    print(f"Generated : {c}")
    print(f'examples : {len(data)}')
    out_dir = source_ent_filepath.replace('.json', '_newly_generated.json')
    with open(out_dir, 'w') as f:
        json.dump(generated_prime, f, indent=1)


    # ---- save token counts in same dir as out_dir ----
    token_file = source_ent_filepath.replace('.json', '_token_counts.txt')
    with open(token_file, 'w') as f:
        f.write("id\tsystem_tokens\thuman_tokens\ttotal_input_tokens\toutput_tokens\n")
        for s in gen.token_stats:
            f.write(f"{s['id']}\t{s['system_tokens']}\t{s['human_tokens']}\t{s['total_tokens']}\t{s['output_tokens']}\n")
        n = len(gen.token_stats)
        if n:
            f.write("\n--- Summary ---\n")
            f.write(f"count\t{n}\n")
            f.write(f"sum_system_tokens\t{gen.tot_sys}\n")
            f.write(f"sum_human_tokens\t{gen.tot_hum}\n")
            f.write(f"sum_total_tokens\t{gen.tot_all}\n")
            f.write(f"sum_output_tokens\t{gen.tot_out}\n")
            f.write(f"avg_system_tokens\t{gen.tot_sys / n:.2f}\n")
            f.write(f"avg_human_tokens\t{gen.tot_hum / n:.2f}\n")
            f.write(f"avg_total_tokens\t{gen.tot_all / n:.2f}\n")
            f.write(f"avg_output_tokens\t{gen.tot_out / n:.2f}\n")


            f.write(f"\nExecution time: {human_readable}\n")
    print(f"Token counts saved to : {token_file}")

if __name__ == '__main__':

    corpus_name=CONFIG['world']
    onto_name=CONFIG['kb_name']
    onto_filepath = f'{CONFIG["data_dir"]}/{CONFIG["kb_file"]}'

    generate_alias_from_entity(corpus_name,onto_name, 
                               source_ent_filepath=f'{CONFIG["data_dir"]}/blink_format/train/(m1_e1)/{CONFIG["exact_match_file"]}', onto_filepath=onto_filepath)

    generate_alias_from_entity(corpus_name,onto_name, 
                               source_ent_filepath=f'{CONFIG["data_dir"]}/blink_format/train/original_data/{CONFIG["biencoder_top1_file"]}', onto_filepath=onto_filepath) 
