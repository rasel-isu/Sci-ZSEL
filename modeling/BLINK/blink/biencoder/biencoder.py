# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig
from transformers import BertTokenizer, RobertaTokenizer
from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from utils import MEDICGraph, MESHGraph, get_category, get_connected_ents_for_the_label

def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        print(f'n_gpu  : {self.n_gpu }')

        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        # blink_base_model_path = params.get("blink_base_model_path", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        with open(f'{params["kb_file_path"]}') as f:
            self.ontology = json.load(f)

        self.set_onto_graph()

        with open(f'{params["data_path"]}/id_map.json') as f:
            self.map_int_to_kb = json.load(f)
            self.map_kb_to_int = {v: k for k, v in self.map_int_to_kb.items()}

    def set_train_samples(self, train_samples):
        self.train_samples_dict = {}
        for i in train_samples:
            cat = get_category(i['mention'], i['label_title'])
            i['category'] = cat
            self.train_samples_dict[i['sample_id']] = i

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores
        
    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None, gt_kb_id=None, sample_ids=None):
        flag = label_input is None
        # scores = self.score_candidate(context_input, cand_input, flag)
        bs = context_input.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            # loss_ = F.cross_entropy(scores, target, reduction="mean")
            # scores_dedup_gt = self.dedup_gt_scores(scores, gt_kb_id)
            # loss = F.cross_entropy(scores_dedup_gt, target, reduction="mean")
            scores, loss, per_sample_loss, diag = self.select_neg_pos_example(gt_kb_id, sample_ids, context_input)

        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)

        

        # gt_kb_id_list = gt_kb_id.tolist()
        # scores_dedup_gt = scores_dedup_gt.tolist()

        # cand_texts = self.tokenizer.batch_decode(cand_input.tolist(), skip_special_tokens=False)
        # context_texts = self.tokenizer.batch_decode(context_input.tolist(), skip_special_tokens=False)
        # scores_list = scores.tolist()

        return loss, scores, diag
    

    def dedup_gt_scores(self,scores, gt_kb_id):
        # gt_kb_id_list looks like [[299], [5632], [5632], ...]; flatten to [bs]
        # gt = torch.tensor([x[0] for x in gt_kb_id_list], device=scores.device)  # [bs]
        gt = gt_kb_id.to(self.device).view(-1)  
        bs = scores.size(0)

        # same[i, j] = True iff row i and column j share the same gold entity
        same = gt.unsqueeze(0) == gt.unsqueeze(1)                               # [bs, bs]
        eye  = torch.eye(bs, dtype=torch.bool, device=scores.device)

        # "keep" mask: keep column j for row i if it's row i's own gold (diagonal),
        # OR it's a column whose gold differs from row i's gold.
        keep = eye | ~same                                                      # [bs, bs]

        # scores_dedup_gt: duplicates (non-diagonal same-entity columns) set to -inf
        NEG = torch.finfo(scores.dtype).min
        scores_dedup_gt = scores.masked_fill(~keep, NEG)
        # scores_dedup_gt = scores.masked_fill(~keep, float('-inf'))              # [bs, bs]
        return scores_dedup_gt
    
    def get_neg_samples_baseline_loss(self,bs, gt, scores, device, sample_ids):
        per_sample_loss = []
        # diagnostics
        dupe_counts = []
        pos_counts = []
        neg_avg = []
        gold_scores = []
        details = []
        for i in range(bs):
            gt_label_id = gt[i].item()
            # indices to keep for row i: the diagonal, plus all j != i with a different gold
            # keep_idx = [j for j in range(bs) if j == i or gt[j].item() != gt_label_id]
            # keep_idx = torch.tensor(keep_idx, device=device, dtype=torch.long)
            # new_target = (keep_idx == i).nonzero(as_tuple=True)[0] 

            keep_idx, new_target, n_dup_pos, n_dup_neg, n_neighbor_drop = self._build_keep_idx(i, bs, gt, gt_label_id, device, neighbors_set=None)

            neighbors_int_id = get_connected_ents_for_the_label(
                    self.onto_graph, self.map_int_to_kb, self.map_kb_to_int, 
                    gt_label_id, include_siblings=True)
            neighbors_set = set(neighbors_int_id)
            sample_id = sample_ids[i].item()
            the_sample =  self.train_samples_dict[sample_id]
            category = the_sample ['category']
            sample_details = self.get_sample_details(the_sample,neighbors_set, keep_idx, new_target, gt)
            details.append(sample_details)

            row_scores = scores[i].index_select(0, keep_idx)  # [K_i], K_i <= bs
            total_neg = row_scores.shape[0]-1
            # row_scores_list = row_scores.tolist()
            # row i's gold was at column i in the original matrix; find where it landed
            used_pos = new_target.shape[0]
            # cross-entropy for a single row: add batch dim, then squeeze
            row_loss = F.cross_entropy(
                row_scores.unsqueeze(0),   # [1, K_i]
                new_target,                # [1]
                reduction="none",
            ).squeeze(0)

            per_sample_loss.append(row_loss)

            # --- diagnostics ---
            pos_counts.append(used_pos)
            neg_avg.append(total_neg)
            n_times = (gt == gt_label_id).sum().item() 
            dupe_counts.append(n_times-1)
            gold_scores.append(row_scores[new_target.item()].item())

        diag = {
            "avg_pos_per_sample": sum(pos_counts) / len(pos_counts) if pos_counts else 0.0,
            "avg_neg_per_sample": sum(neg_avg) / len(neg_avg) if neg_avg else 0.0,
            "avg_dupes_per_sample": sum(dupe_counts) / len(dupe_counts) if dupe_counts else 0.0,
            "max_dupes_per_sample": max(dupe_counts) if dupe_counts else 0,
            "samples_with_dupes": sum(1 for d in dupe_counts if d > 0),
            "avg_gold_diag_score": sum(gold_scores) / len(gold_scores) if gold_scores else 0.0,
            "diagonal_is_gold_pct": 1.0,
            'sample_details':details
        }
        return per_sample_loss, diag
    
    def get_baseline_entire_kb_neg_loss(self, bs, gt, device, sample_ids, context_input):
        # ---- 1. Unique batch golds (preserve first-occurrence order for row->col mapping) ----
        n_kb = self.candidate_pool.size(0)

        gt_list = gt.tolist()
        unique_golds = []
        gold_to_col = {}
        for kb_id in gt_list:
            if kb_id not in gold_to_col:
                gold_to_col[kb_id] = len(unique_golds)
                unique_golds.append(kb_id)
        n_unique_golds = len(unique_golds)
        gold_set = set(unique_golds)

        # ---- 2. Sample random KB negs to fill remaining budget (total candidates == bs) ----
        n_random = bs - n_unique_golds  # could be 0 if every row has a unique gold (rare)
        neg_ids = []
        if n_random > 0:
            seen = set()
            while len(neg_ids) < n_random:
                cand = torch.randint(0, n_kb, (n_random * 2,), device=device).tolist()
                for c in cand:
                    if c in gold_set or c in seen:
                        continue
                    seen.add(c)
                    neg_ids.append(c)
                    if len(neg_ids) == n_random:
                        break

        all_ids = unique_golds + neg_ids                                    # len == bs
        all_ids_t = torch.tensor(all_ids, device=device, dtype=torch.long)  # [bs]

        # ---- 3. Single forward pass, exactly bs candidates ----
        combined_cands = self.candidate_pool.index_select(0, all_ids_t)     # [bs, max_len]
        scores = self.score_candidate(context_input, combined_cands, True)  # [bs, bs]

        # ---- 4. Per-row loss ----
        per_sample_loss = []
        dupe_counts, pos_counts, neg_avg, gold_scores, details = [], [], [], [], []

        neg_ids_t = torch.tensor(neg_ids, device=device, dtype=torch.long) if neg_ids \
                    else torch.empty(0, device=device, dtype=torch.long)
        neg_col_offset = n_unique_golds

        for i in range(bs):
            gt_label_id = gt[i].item()
            pos_col = gold_to_col[gt_label_id]

            # gold negs: every unique-gold column except this row's
            gold_neg_cols = [c for kb_id, c in gold_to_col.items() if kb_id != gt_label_id]
            gold_neg_cols_t = torch.tensor(gold_neg_cols, device=device, dtype=torch.long)

            # random negs: defensive collision check (shouldn't trigger; gold_set filter prevents it)
            if neg_ids_t.numel() > 0:
                neg_keep_mask = neg_ids_t != gt_label_id
                rand_neg_cols = (torch.arange(neg_ids_t.shape[0], device=device) + neg_col_offset)[neg_keep_mask]
            else:
                rand_neg_cols = torch.empty(0, device=device, dtype=torch.long)

            pos_col_t = torch.tensor([pos_col], device=device, dtype=torch.long)
            keep_idx_full = torch.cat([pos_col_t, gold_neg_cols_t, rand_neg_cols], dim=0)
            new_target = torch.tensor([0], device=device, dtype=torch.long)

            row_scores = scores[i].index_select(0, keep_idx_full)

            # diagnostics
            neighbors_int_id = get_connected_ents_for_the_label(
                self.onto_graph, self.map_int_to_kb, self.map_kb_to_int,
                gt_label_id, include_siblings=True)
            neighbors_set = set(neighbors_int_id)
            the_sample = self.train_samples_dict[sample_ids[i].item()]
            keep_idx_kb_ids = all_ids_t.index_select(0, keep_idx_full)
            details.append(self.get_sample_details_kb_based(the_sample, neighbors_set, keep_idx_kb_ids, new_target))

            total_neg = row_scores.shape[0] - 1
            used_pos  = new_target.shape[0]

            row_loss = F.cross_entropy(
                row_scores.unsqueeze(0),
                new_target,
                reduction="none",
            ).squeeze(0)
            per_sample_loss.append(row_loss)

            pos_counts.append(used_pos)
            neg_avg.append(total_neg)
            n_times = (gt == gt_label_id).sum().item()
            dupe_counts.append(n_times - 1)
            gold_scores.append(row_scores[0].item())

        diag = self._build_diag(
                pos_counts, neg_avg, dupe_counts, gold_scores,
                n_unique_golds, len(neg_ids), len(all_ids), details
            )
        return scores, per_sample_loss, diag
    
    def get_non_neighbors_entire_kb_loss(self, bs, gt, device, sample_ids, context_input, consider_siblings):
        # ---- 1. Unique batch golds ----
        n_kb = self.candidate_pool.size(0)

        gt_list = gt.tolist()
        unique_golds = []
        gold_to_col = {}
        for kb_id in gt_list:
            if kb_id not in gold_to_col:
                gold_to_col[kb_id] = len(unique_golds)
                unique_golds.append(kb_id)
        n_unique_golds = len(unique_golds)
        gold_set = set(unique_golds)

        # ---- 2. Per-row neighbor sets ----
        row_neighbor_sets = []
        max_row_neighbors = 0
        for i in range(bs):
            nb = set(get_connected_ents_for_the_label(
                self.onto_graph, self.map_int_to_kb, self.map_kb_to_int,
                gt[i].item(), include_siblings=consider_siblings))
            row_neighbor_sets.append(nb)
            if len(nb) > max_row_neighbors:
                max_row_neighbors = len(nb)

        # ---- 3. Sample random KB negs with headroom for per-row neighbor filtering ----
        HEADROOM_CAP = 30  # tune; 0 = identical memory to baseline_entire_kb
        headroom = min(max_row_neighbors, HEADROOM_CAP)
        n_random_target = (bs - n_unique_golds) + headroom
        available = n_kb - len(gold_set)
        n_random_target = min(n_random_target, available)

        neg_ids = []
        if n_random_target > 0:
            seen = set()
            while len(neg_ids) < n_random_target:
                cand = torch.randint(0, n_kb, (n_random_target * 2,), device=device).tolist()
                for c in cand:
                    if c in gold_set or c in seen:
                        continue
                    seen.add(c)
                    neg_ids.append(c)
                    if len(neg_ids) == n_random_target:
                        break

        all_ids = unique_golds + neg_ids
        all_ids_t = torch.tensor(all_ids, device=device, dtype=torch.long)  # column -> KB id

        # ---- 4. Single forward pass ----
        combined_cands = self.candidate_pool.index_select(0, all_ids_t)
        scores = self.score_candidate(context_input, combined_cands, True)

        # ---- 5. Per-row loss with neighbor filtering ----
        per_sample_loss = []
        dupe_counts, pos_counts, neg_avg, gold_scores, details = [], [], [], [], []

        neg_ids_list = neg_ids
        n_neg_t = len(neg_ids_list)
        neg_col_offset = n_unique_golds
        target_n_random_per_row = bs - n_unique_golds

        for i in range(bs):
            gt_label_id = gt[i].item()
            pos_col = gold_to_col[gt_label_id]
            neighbors_set = row_neighbor_sets[i]

            gold_neg_cols = [c for kb_id, c in gold_to_col.items()
                            if kb_id != gt_label_id and kb_id not in neighbors_set]
            gold_neg_cols_t = torch.tensor(gold_neg_cols, device=device, dtype=torch.long)

            # how many gold negs we lost vs. unfiltered case (n_unique_golds - 1)
            gold_lost = (n_unique_golds - 1) - len(gold_neg_cols)

            # random negs: target = original target + gold loss top-up
            row_random_target = target_n_random_per_row + gold_lost

            if n_neg_t > 0:
                keep_positions = [
                    p for p, nid in enumerate(neg_ids_list)
                    if (nid not in neighbors_set) and (nid != gt_label_id)
                ]
                keep_positions = keep_positions[:row_random_target]   # use bumped target
                rand_neg_cols = torch.tensor(keep_positions, device=device, dtype=torch.long) + neg_col_offset
            else:
                rand_neg_cols = torch.empty(0, device=device, dtype=torch.long)

            pos_col_t = torch.tensor([pos_col], device=device, dtype=torch.long)
            keep_idx_full = torch.cat([pos_col_t, gold_neg_cols_t, rand_neg_cols], dim=0)
            new_target = torch.tensor([0], device=device, dtype=torch.long)

            row_scores = scores[i].index_select(0, keep_idx_full)

            # diagnostics — translate column indices to KB ids
            keep_idx_kb_ids = all_ids_t.index_select(0, keep_idx_full)
            the_sample = self.train_samples_dict[sample_ids[i].item()]
            details.append(self.get_sample_details_kb_based(
                the_sample, neighbors_set, keep_idx_kb_ids, new_target
            ))

            total_neg = row_scores.shape[0] - 1
            used_pos  = new_target.shape[0]

            row_loss = F.cross_entropy(
                row_scores.unsqueeze(0),
                new_target,
                reduction="none",
            ).squeeze(0)
            per_sample_loss.append(row_loss)

            pos_counts.append(used_pos)
            neg_avg.append(total_neg)
            n_times = (gt == gt_label_id).sum().item()
            dupe_counts.append(n_times - 1)
            gold_scores.append(row_scores[0].item())

        diag = self._build_diag(
                pos_counts, neg_avg, dupe_counts, gold_scores,
                n_unique_golds, n_neg_t, len(all_ids), details,
                # # function-specific extras (only for non_neighbors version):
                # max_row_neighbors=max_row_neighbors
            )
        return scores, per_sample_loss, diag

    def _build_diag(self, pos_counts, neg_avg, dupe_counts, gold_scores,
                n_unique_golds, n_neg_t, n_total_cands, details, **extras):
        diag = {
            "avg_pos_per_sample":   sum(pos_counts) / len(pos_counts) if pos_counts else 0.0,
            "avg_neg_per_sample":   sum(neg_avg) / len(neg_avg) if neg_avg else 0.0,
            "avg_dupes_per_sample": sum(dupe_counts) / len(dupe_counts) if dupe_counts else 0.0,
            "max_dupes_per_sample": max(dupe_counts) if dupe_counts else 0,
            "samples_with_dupes":   sum(1 for d in dupe_counts if d > 0),
            "avg_gold_diag_score":  sum(gold_scores) / len(gold_scores) if gold_scores else 0.0,
            "diagonal_is_gold_pct": 1.0,
            "n_unique_golds":       n_unique_golds,
            "n_kb_neg_sampled":     n_neg_t,
            "n_total_cands":        n_total_cands,
            "total_neg_count":      sum(neg_avg),
            "total_pos_count":      sum(pos_counts),
            "total_dupe_count":     sum(dupe_counts),
            "total_gold_score":     sum(gold_scores),
            "total_rows":           len(neg_avg),
            "sample_details":       details,
        }
        diag.update(extras)
        return diag

    def get_neg_samples_masked_baseline_loss(self, bs, gt, scores, device):
        """
        Vectorized: mask duplicate-gold columns (j != i with same gold) 
        from denominator. Keep column i as the positive anchor.
        """
        NEG_INF = torch.finfo(scores.dtype).min

        # Build mask: [bs, bs], mask[i, j] = True iff j != i AND gt[j] == gt[i]
        gt_row = gt.view(-1, 1)      # [bs, 1]
        gt_col = gt.view(1, -1)      # [1, bs]
        same_gold = (gt_row == gt_col)                          # [bs, bs]
        not_diagonal = ~torch.eye(bs, dtype=torch.bool, device=device)
        mask = same_gold & not_diagonal                         # [bs, bs]

        # Apply mask to scores
        masked_scores = scores.masked_fill(mask, NEG_INF)       # [bs, bs]

        # Target is diagonal (column i is positive for row i)
        target = torch.arange(bs, device=device, dtype=torch.long)

        # Single vectorized cross-entropy
        per_sample_loss = F.cross_entropy(
            masked_scores, target, reduction="none"
        )  # [bs]

        # --- diagnostics (vectorized) ---
        with torch.no_grad():
            dupe_counts = mask.sum(dim=1)                       # [bs]
            diag_scores = scores.diagonal()                     # [bs]

            diag = {
                "avg_dupes_per_sample": dupe_counts.float().mean().item(),
                "max_dupes_per_sample": int(dupe_counts.max().item()),
                "samples_with_dupes": int((dupe_counts > 0).sum().item()),
                "avg_gold_diag_score": diag_scores.mean().item(),
                "diagonal_is_gold_pct": 1.0,
            }

        return per_sample_loss, diag

    def get_neg_samples_non_neighbors_loss(self,bs, gt, scores, device, sample_ids, consider_siblings):
        per_sample_loss = []
        # diagnostics
        dupe_counts = []
        pos_counts = []
        neg_avg = []
        details = []
        gold_scores = []
        for i in range(bs):
            gt_label_id = gt[i].item()
            sample_id = sample_ids[i].item()

            neighbors_int_id = get_connected_ents_for_the_label(
                self.onto_graph, self.map_int_to_kb, self.map_kb_to_int, 
                gt_label_id, include_siblings=consider_siblings)
            neighbors_set = set(neighbors_int_id)

            
            the_sample =  self.train_samples_dict[sample_id]
            category = the_sample['category']

            # keep_idx = [j for j in range(bs)
            #             if j == i or (gt[j].item() != gt_label_id and gt[j].item() not in neighbors_set)]
            # keep_idx = torch.tensor(keep_idx, device=device, dtype=torch.long)
            # new_target = (keep_idx == i).nonzero(as_tuple=True)[0]  # scalar tensor


            keep_idx, new_target, n_dup_pos, n_dup_neg, n_nb = self._build_keep_idx(
            i, bs, gt, gt_label_id, device, neighbors_set=neighbors_set)

            sample_details = self.get_sample_details(the_sample,neighbors_set, keep_idx, new_target, gt)
            details.append(sample_details)

            row_scores = scores[i].index_select(0, keep_idx)  # [K_i], K_i <= bs
            total_neg = row_scores.shape[0]-1
            # row_scores_list = row_scores.tolist()
            # row i's gold was at column i in the original matrix; find where it landed
            used_pos = new_target.shape[0]

            # cross-entropy for a single row: add batch dim, then squeeze
            row_loss = F.cross_entropy(
                row_scores.unsqueeze(0),   # [1, K_i]
                new_target,                # [1]
                reduction="none",
            ).squeeze(0)
            per_sample_loss.append(row_loss)

            # --- diagnostics ---
            pos_counts.append(used_pos)
            neg_avg.append(total_neg)
            n_times = (gt == gt_label_id).sum().item() 
            dupe_counts.append(n_times-1)
            gold_scores.append(row_scores[new_target.item()].item())

        diag = {
            "avg_pos_per_sample": sum(pos_counts) / len(pos_counts) if pos_counts else 0.0,
            "avg_neg_per_sample": sum(neg_avg) / len(neg_avg) if neg_avg else 0.0,
            "avg_dupes_per_sample": sum(dupe_counts) / len(dupe_counts) if dupe_counts else 0.0,
            "max_dupes_per_sample": max(dupe_counts) if dupe_counts else 0,
            "samples_with_dupes": sum(1 for d in dupe_counts if d > 0),
            "avg_gold_diag_score": sum(gold_scores) / len(gold_scores) if gold_scores else 0.0,
            "diagonal_is_gold_pct": 1.0,
            'sample_details':details
        }
        return per_sample_loss, diag
    
    def get_loss_for_pcs_base_combo(self,bs, gt, scores, device, sample_ids, pcs_categories, consider_siblings=True):
        per_sample_loss = []
        # diagnostics
        dupe_counts = []
        pos_counts = []
        neg_avg = []
        gold_scores = []
        details = []
        for i in range(bs):
            gt_label_id = gt[i].item()
            sample_id = sample_ids[i].item()

            neighbors_int_id = get_connected_ents_for_the_label(
                    self.onto_graph, self.map_int_to_kb, self.map_kb_to_int, 
                    gt_label_id, include_siblings=consider_siblings)
            neighbors_set = set(neighbors_int_id)
            
            the_sample =  self.train_samples_dict[sample_id]
            category = the_sample ['category']
            # if category in pcs_categories:
            #     keep_idx = [j for j in range(bs)
            #                 if j == i or (gt[j].item() != gt_label_id and gt[j].item() not in neighbors_set)]
            # else:
            #     keep_idx = [j for j in range(bs) if j == i or gt[j].item() != gt_label_id]

            # keep_idx = torch.tensor(keep_idx, device=device, dtype=torch.long)
            # new_target = (keep_idx == i).nonzero(as_tuple=True)[0]  # scalar tensor

            ns = neighbors_set if category in pcs_categories else None
            keep_idx, new_target, n_dup_pos, n_dup_neg, n_nb = self._build_keep_idx(
                i, bs, gt, gt_label_id, device, neighbors_set=ns)

            sample_details = self.get_sample_details(the_sample,neighbors_set, keep_idx, new_target, gt)
            details.append(sample_details)

            row_scores = scores[i].index_select(0, keep_idx)  # [K_i], K_i <= bs
            total_neg = row_scores.shape[0]-1
            # row_scores_list = row_scores.tolist()
            # row i's gold was at column i in the original matrix; find where it landed
            used_pos = new_target.shape[0]

            # cross-entropy for a single row: add batch dim, then squeeze
            row_loss = F.cross_entropy(
                row_scores.unsqueeze(0),   # [1, K_i]
                new_target,                # [1]
                reduction="none",
            ).squeeze(0)
            per_sample_loss.append(row_loss)

            # --- diagnostics ---
            pos_counts.append(used_pos)
            neg_avg.append(total_neg)
            n_times = (gt == gt_label_id).sum().item() 
            dupe_counts.append(n_times-1)
            gold_scores.append(row_scores[new_target.item()].item())

        diag = {
            "avg_pos_per_sample": sum(pos_counts) / len(pos_counts) if pos_counts else 0.0,
            "avg_neg_per_sample": sum(neg_avg) / len(neg_avg) if neg_avg else 0.0,
            "avg_dupes_per_sample": sum(dupe_counts) / len(dupe_counts) if dupe_counts else 0.0,
            "max_dupes_per_sample": max(dupe_counts) if dupe_counts else 0,
            "samples_with_dupes": sum(1 for d in dupe_counts if d > 0),
            "avg_gold_diag_score": sum(gold_scores) / len(gold_scores) if gold_scores else 0.0,
            "diagonal_is_gold_pct": 1.0,
            'sample_details':details
        }
        
        return per_sample_loss, diag

    def get_sample_details(self, the_sample, neighbors_set, keep_idx, target, gt):
        base_info = f"id:{the_sample['sample_id']}|m:{the_sample['mention']}|t:{the_sample['label_title']}|c:{the_sample['category']}"
        neighbors = ' | '.join([ self.ontology[self.map_int_to_kb[str(int_id)]]['name'] for int_id in neighbors_set])
        selected = gt.index_select(0, keep_idx)
        target_pos = target.item()
        selected = torch.cat([selected[:target_pos], selected[target_pos + 1:]])
        selected_negs = ' | '.join([ self.ontology[self.map_int_to_kb[str(int_id.item())]]['name'] for int_id in  selected])
        details = {'sample':base_info, 'neighbors':neighbors, 'negatives':selected_negs}
        return details

    def get_sample_details_kb_based(self, the_sample, neighbors_set, keep_idx_kb_ids, target):
        """keep_idx_kb_ids: 1-D long tensor of KB ids in keep_idx order (gold first, then negs).
                            target's value is the position of the gold within keep_idx_kb_ids.
        Pass KB ids directly from the caller; no more gt.index_select inside."""
        base_info = f"id:{the_sample['sample_id']}|m:{the_sample['mention']}|t:{the_sample['label_title']}|c:{the_sample['category']}"
        neighbors = ' | '.join([
            self.ontology[self.map_int_to_kb[str(int_id)]]['name']
            for int_id in neighbors_set
        ])
        target_pos = target.item()
        # remove the gold position to get only the negative KB ids
        selected_negs_t = torch.cat([keep_idx_kb_ids[:target_pos], keep_idx_kb_ids[target_pos + 1:]])
        selected_negs = ' | '.join([
            self.ontology[self.map_int_to_kb[str(int_id.item())]]['name']
            for int_id in selected_negs_t
        ])
        return {'sample': base_info, 'neighbors': neighbors, 'negatives': selected_negs}

    def select_neg_pos_example(self, gt_kb_id, sample_ids, context_input):
        bs = context_input.size(0)
        device = self.device
        gt = gt_kb_id.to(device).view(-1)  # [bs]

        diag = {"avg_pos_per_sample":0.0,"avg_neg_per_sample":0.0, "avg_dupes_per_sample": 0.0,"max_dupes_per_sample": 0,"samples_with_dupes": 0,
        "avg_gold_diag_score": 0.0,"diagonal_is_gold_pct": 0.0,'sample_details':''}

        # scores_list = scores.tolist()
        if self.params['bi_enc_negative_selection'] == 'add_prch_in_pos_list':
            # per_sample_loss, diag = self.get_multi_positive_neighbors_loss(bs, gt, scores, device)
            scores, per_sample_loss, diag = self.get_multi_positive_neighbors_entire_kb_loss(bs, gt, device, sample_ids, context_input, consider_siblings=False)
        elif self.params['bi_enc_negative_selection'] == 'remove_prch_from_neg_list':
            # per_sample_loss, diag = self.get_neg_samples_non_neighbors_loss(bs, gt, scores, device,sample_ids, consider_siblings=False)
            scores, per_sample_loss, diag = self.get_non_neighbors_entire_kb_loss(bs, gt, device, sample_ids, context_input, consider_siblings=False)
        elif self.params['bi_enc_negative_selection'] == 'remove_prchsbl_from_neg_list':
            # per_sample_loss, diag = self.get_neg_samples_non_neighbors_loss(bs, gt, scores, device, sample_ids, consider_siblings=True)
            scores, per_sample_loss, diag = self.get_non_neighbors_entire_kb_loss(bs, gt, device, sample_ids, context_input, consider_siblings=True)
        elif self.params['bi_enc_negative_selection'] == 'RM-PCS-NO':
            per_sample_loss, diag = self.get_loss_for_pcs_base_combo(bs, gt, scores, device, sample_ids, pcs_categories=['NO'])
        elif self.params['bi_enc_negative_selection'] == 'RM-PCS-NO-LO':
            per_sample_loss, diag = self.get_loss_for_pcs_base_combo(bs, gt, scores, device, sample_ids, pcs_categories=['NO', 'LO'])
        elif self.params['bi_enc_negative_selection'] == 'BASE':
            per_sample_loss, diag = self.get_neg_samples_baseline_loss(bs, gt, scores, device, sample_ids)
            # per_sample_loss, diag = self.get_neg_samples_masked_baseline_loss(bs, gt, scores, device)
        elif self.params['bi_enc_negative_selection'] == 'BASE-NEG-ENTIRE-KB':
            scores, per_sample_loss, diag = self.get_baseline_entire_kb_neg_loss(bs, gt, device, sample_ids, context_input)
        


        per_sample_loss = torch.stack(per_sample_loss)  # [bs]
        loss = per_sample_loss.mean()

        return scores, loss, per_sample_loss, diag

    def get_multi_positive_neighbors_loss(self, bs, gt, scores, device):
        per_sample_loss = []
        dupe_counts, pos_counts, neg_avg, gold_scores = [], [], [], []

        for i in range(bs):
            gt_label_id = gt[i].item()
            neighbors_int_id = get_connected_ents_for_the_label(
                self.onto_graph, self.map_int_to_kb, self.map_kb_to_int,
                gt_label_id, include_siblings=False
            )
            neighbors_set = set(neighbors_int_id)

            # ---- partition columns into positives vs negatives ----
            positive_mask = torch.zeros(bs, dtype=torch.bool, device=device)
            positive_mask[i] = True
            for j in range(bs):
                if j != i and gt[j].item() in neighbors_set:
                    positive_mask[j] = True

            if positive_mask.sum().item() == bs:
                continue  # no negatives, skip

            # ---- dedup positives by gold int id (keep first occurrence) ----
            # diagonal i is always kept as a positive representative for gt_label_id
            seen_pos_golds = {gt_label_id}
            unique_positive_cols = [i]
            for j in positive_mask.nonzero(as_tuple=True)[0].tolist():
                if j == i:
                    continue
                j_gold = gt[j].item()
                if j_gold in seen_pos_golds:
                    continue
                seen_pos_golds.add(j_gold)
                unique_positive_cols.append(j)

            # ---- dedup negatives by gold int id (keep first occurrence) ----
            seen_neg_golds = set()
            unique_negative_cols = []
            for j in (~positive_mask).nonzero(as_tuple=True)[0].tolist():
                j_gold = gt[j].item()
                if j_gold in seen_neg_golds:
                    continue
                seen_neg_golds.add(j_gold)
                unique_negative_cols.append(j)

            if not unique_negative_cols:
                continue

            neg_idx = torch.tensor(unique_negative_cols, device=device, dtype=torch.long)
            neg_scores = scores[i].index_select(0, neg_idx)  # [n_neg]

            # ---- per-positive CE: positive at index 0, then unique negatives ----
            row_losses = []
            for p in unique_positive_cols:
                p_score = scores[i, p].unsqueeze(0)               # [1]
                row_scores = torch.cat([p_score, neg_scores])     # [1 + n_neg]
                target = torch.zeros(1, device=device, dtype=torch.long)
                p_loss = F.cross_entropy(row_scores.unsqueeze(0), target, reduction="none").squeeze(0)
                row_losses.append(p_loss)

            row_loss = torch.stack(row_losses).mean()
            per_sample_loss.append(row_loss)

            # ---- diagnostics ----
            pos_counts.append(len(unique_positive_cols))
            neg_avg.append(len(unique_negative_cols))
            n_dupes = sum(1 for j in range(bs) if j != i and gt[j].item() == gt_label_id)
            dupe_counts.append(n_dupes)
            gold_scores.append(scores[i, i].item())

        diag = {
            "avg_pos_per_sample": sum(pos_counts) / len(pos_counts) if pos_counts else 0.0,
            "avg_neg_per_sample": sum(neg_avg) / len(neg_avg) if neg_avg else 0.0,
            "avg_dupes_per_sample": sum(dupe_counts) / len(dupe_counts) if dupe_counts else 0.0,
            "max_dupes_per_sample": max(dupe_counts) if dupe_counts else 0,
            "samples_with_dupes": sum(1 for d in dupe_counts if d > 0),
            "avg_gold_diag_score": sum(gold_scores) / len(gold_scores) if gold_scores else 0.0,
            "diagonal_is_gold_pct": 1.0,
        }
        return per_sample_loss, diag

    def get_multi_positive_neighbors_entire_kb_loss(self, bs, gt, device, sample_ids, context_input, consider_siblings):
        # ---- 1. Unique batch golds (preserve first-occurrence order for row->col mapping) ----
        n_kb = self.candidate_pool.size(0)

        gt_list = gt.tolist()
        unique_golds = []
        gold_to_col = {}
        for kb_id in gt_list:
            if kb_id not in gold_to_col:
                gold_to_col[kb_id] = len(unique_golds)
                unique_golds.append(kb_id)
        n_unique_golds = len(unique_golds)
        gold_set = set(unique_golds)

        # ---- 2. Per-row neighbor sets ----
        row_neighbor_sets = []
        max_row_neighbors = 0
        for i in range(bs):
            nb = set(get_connected_ents_for_the_label(
                self.onto_graph, self.map_int_to_kb, self.map_kb_to_int,
                gt[i].item(), include_siblings=consider_siblings))
            row_neighbor_sets.append(nb)
            if len(nb) > max_row_neighbors:
                max_row_neighbors = len(nb)

        # ---- 3. Sample random KB negs (same headroom strategy as non_neighbors version) ----
        HEADROOM_CAP = 30  # tune; 0 = identical memory to baseline_entire_kb
        headroom = min(max_row_neighbors, HEADROOM_CAP)
        n_random_target = (bs - n_unique_golds) + headroom
        available = n_kb - len(gold_set)
        n_random_target = min(n_random_target, available)

        neg_ids = []
        if n_random_target > 0:
            seen = set()
            while len(neg_ids) < n_random_target:
                cand = torch.randint(0, n_kb, (n_random_target * 2,), device=device).tolist()
                for c in cand:
                    if c in gold_set or c in seen:
                        continue
                    seen.add(c)
                    neg_ids.append(c)
                    if len(neg_ids) == n_random_target:
                        break

        all_ids = unique_golds + neg_ids
        all_ids_t = torch.tensor(all_ids, device=device, dtype=torch.long)  # column -> KB id

        # ---- 4. Single forward pass ----
        combined_cands = self.candidate_pool.index_select(0, all_ids_t)
        scores = self.score_candidate(context_input, combined_cands, True)
        # scores shape: [bs, n_unique_golds + n_random_target]

        # ---- 5. Per-row multi-positive loss ----
        per_sample_loss = []
        dupe_counts, pos_counts, neg_avg, gold_scores, details = [], [], [], [], []

        neg_ids_list = neg_ids
        n_neg_t = len(neg_ids_list)
        neg_col_offset = n_unique_golds
        target_n_random_per_row = bs - n_unique_golds

        for i in range(bs):
            gt_label_id = gt[i].item()
            pos_col = gold_to_col[gt_label_id]
            neighbors_set = row_neighbor_sets[i]

            # ---- positive columns: this row's gold + any other unique gold that is a neighbor ----
            positive_cols = [pos_col]
            for kb_id, c in gold_to_col.items():
                if kb_id != gt_label_id and kb_id in neighbors_set:
                    positive_cols.append(c)
            # Already deduplicated by construction (unique_golds is a deduped list).

            # ---- negative columns from gold side: unique golds that are neither this row's gold nor a neighbor ----
            gold_neg_cols = [c for kb_id, c in gold_to_col.items()
                            if kb_id != gt_label_id and kb_id not in neighbors_set]

            # ---- negative columns from random KB pool: drop neighbors + defensive gold collision ----
            gold_lost = (n_unique_golds - 1) - len(gold_neg_cols)  # gold negs lost to neighbor filter
            row_random_target = target_n_random_per_row + gold_lost  # top up from random pool

            if n_neg_t > 0:
                keep_positions = [
                    p for p, nid in enumerate(neg_ids_list)
                    if (nid not in neighbors_set) and (nid != gt_label_id)
                ]
                keep_positions = keep_positions[:row_random_target]
                rand_neg_cols = [p + neg_col_offset for p in keep_positions]
            else:
                rand_neg_cols = []

            neg_cols_all = gold_neg_cols + rand_neg_cols
            if not neg_cols_all:
                continue  # no negatives, skip this row (matches original behavior)

            neg_idx = torch.tensor(neg_cols_all, device=device, dtype=torch.long)
            neg_scores = scores[i].index_select(0, neg_idx)  # [n_neg]

            # ---- per-positive CE: positive at index 0, then negatives ----
            row_losses = []
            for p_col in positive_cols:
                p_score = scores[i, p_col].unsqueeze(0)               # [1]
                row_scores = torch.cat([p_score, neg_scores])         # [1 + n_neg]
                target = torch.zeros(1, device=device, dtype=torch.long)
                p_loss = F.cross_entropy(
                    row_scores.unsqueeze(0), target, reduction="none"
                ).squeeze(0)
                row_losses.append(p_loss)

            row_loss = torch.stack(row_losses).mean()
            per_sample_loss.append(row_loss)

            # ---- diagnostics ----
            pos_counts.append(len(positive_cols))
            neg_avg.append(len(neg_cols_all))
            n_times = (gt == gt_label_id).sum().item()
            dupe_counts.append(n_times - 1)
            gold_scores.append(scores[i, pos_col].item())  # this row's gold score

            # column indices used for this row's loss (positives + negatives) -> KB ids for details
            used_cols = torch.tensor(positive_cols + neg_cols_all, device=device, dtype=torch.long)
            used_kb_ids = all_ids_t.index_select(0, used_cols)
            # target position within `used_cols`: index 0 (this row's gold is the first positive)
            target_for_details = torch.tensor([0], device=device, dtype=torch.long)
            the_sample = self.train_samples_dict[sample_ids[i].item()]
            details.append(self.get_sample_details_kb_based(
                the_sample, neighbors_set, used_kb_ids, target_for_details
            ))

        diag = self._build_diag(
                pos_counts, neg_avg, dupe_counts, gold_scores,
                n_unique_golds, n_neg_t, len(all_ids), details,
                # # function-specific extras (only for non_neighbors version):
                # max_row_neighbors=max_row_neighbors
            )

        return scores, per_sample_loss, diag



    def _build_keep_idx(self, i, bs, gt, gt_label_id, device, neighbors_set=None):
        """
        Build keep_idx for row i of the in-batch score matrix.

        Always:
        - keeps the diagonal (column i)
        - drops duplicate positives (j != i with gt[j] == gt_label_id)
        - drops duplicate negatives (multiple j's sharing the same gt[j])
        Optionally:
        - drops ontology neighbors of gt_label_id when neighbors_set is given

        Returns: (keep_idx tensor, new_target tensor, n_dup_pos, n_dup_neg, n_neighbor_drop)
        """
        seen_neg_golds = set()
        keep_idx = []
        n_dup_pos = n_dup_neg = n_neighbor_drop = 0
        for j in range(bs):
            if j == i:
                keep_idx.append(j)                       # diagonal — always keep
                continue
            j_gold = gt[j].item()
            if j_gold == gt_label_id:                    # duplicate positive
                n_dup_pos += 1
                continue
            if neighbors_set is not None and j_gold in neighbors_set:
                n_neighbor_drop += 1                     # ontology neighbor
                continue
            if j_gold in seen_neg_golds:                 # duplicate negative
                n_dup_neg += 1
                continue
            seen_neg_golds.add(j_gold)
            keep_idx.append(j)

        keep_idx = torch.tensor(keep_idx, device=device, dtype=torch.long)
        new_target = (keep_idx == i).nonzero(as_tuple=True)[0]
        return keep_idx, new_target, n_dup_pos, n_dup_neg, n_neighbor_drop

    def set_onto_graph(self):
        if self.params["onto"] =='ncbi_disease':
            self.onto_graph = MEDICGraph(self.params["kb_file_path"])
        elif self.params["onto"] =='bc5cdr':
            self.onto_graph = MESHGraph()
        elif self.params["onto"]  in ['cmo', 'vt', 'lpt', 'cometa', 'MedMentions']:
            self.onto_graph = MEDICGraph(self.params["kb_file_path"])

    def set_onto_encoding(self):
        self.encoded_kb = torch.load(self.params.get("cand_encode_path", None))
        self.candidate_pool = torch.load(self.params.get("cand_pool_path", None))
        self.candidate_pool = self.candidate_pool.to(self.device)


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
