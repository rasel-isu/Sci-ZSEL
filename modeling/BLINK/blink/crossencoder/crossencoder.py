
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME 
from transformers import (BertModel)
from transformers import (RobertaModel)
from transformers import BertTokenizer, RobertaTokenizer
from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG

def load_crossencoder(params):
    crossencoder = CrossEncoderRanker(params)
    return crossencoder

class CrossEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(CrossEncoderModule, self).__init__()
        model_path = params["bert_model"]
        if params.get("roberta"):
            encoder_model = RobertaModel.from_pretrained(model_path)
        else:
            encoder_model = BertModel.from_pretrained(model_path)
        encoder_model.resize_token_embeddings(len(tokenizer))
        self.encoder = BertEncoder(
            encoder_model,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            dropout_rate=params["dropout_rate"],
        )
        self.config = self.encoder.bert_model.config

    def forward(
        self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
    ):

        embedding_ctxt = self.encoder(token_idx_ctxt, segment_idx_ctxt, mask_ctxt)
        return embedding_ctxt.squeeze(-1)


class CrossEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(CrossEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        if params.get("roberta"):
            self.tokenizer = RobertaTokenizer.from_pretrained(params["bert_model"],)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                params["bert_model"], do_lower_case=params["lowercase"]
            )

        special_tokens_dict = {
            "additional_special_tokens": [
                ENT_START_TAG,
                ENT_END_TAG,
                ENT_TITLE_TAG,
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.NULL_IDX = self.tokenizer.pad_token_id
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token
        
        # init model
        self.build_model()
        if params["path_to_model"] is not None:
            self.load_model(params["path_to_model"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.error_count = 0
        self.prch_added_as_neg = []
        self.epoch = None

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def save(self, output_dir):
        self.save_model(output_dir)
        self.tokenizer.save_vocabulary(output_dir)

    def build_model(self):
        self.model = CrossEncoderModule(self.params, self.tokenizer)
    
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

    def score_candidate(self, text_vecs, context_len):
        # Encode contexts first
        num_cand = text_vecs.size(1)
        text_vecs = text_vecs.view(-1, text_vecs.size(-1))
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX, context_len,
        )

        embedding_ctxt = self.model(token_idx_ctxt, segment_idx_ctxt, mask_ctxt,)

        return embedding_ctxt.view(-1, num_cand)

    def forward(self, input_idx, label_input, context_len, 
                label_input_gt=None, ranking_loss_fn=None, graph_candidates=None, 
                sample_id=None, is_train=False):
        
        if is_train:
            if self.params['cross_enc_negative_selection'] == 'crsenc_ranked_neg_after_label':
                if self.epoch == 0:
                    output_idx, output_labels = self.select_random_negative_candidates(input_idx, label_input, n=20)
                    scores = self.score_candidate(output_idx, context_len)
                    correct_count = self.count_correctly_predicted(scores, output_labels)
                    loss = F.cross_entropy(scores, output_labels, reduction="mean")
                elif self.epoch > 0:
                    # scores = self.score_candidate(input_idx, context_len)
                    # loss = self.get_loss_of_score_from_crsenc_ranked_neg_after_label(scores, label_input, n=20)
                    with torch.no_grad(): 
                        scores = self.score_candidate(input_idx, context_len)
                    loss = self.get_loss_of_input_idx_from_crsenc_ranked_neg_after_label(scores, input_idx, label_input, context_len, n=20)
                    correct_count = self.count_correctly_predicted(scores, label_input)
                return loss, scores, correct_count
            
            elif self.params['cross_enc_negative_selection'] == 'comb_crsenc_ranked_neg_and_prnt_chld_as_neg':
                if self.epoch == 0:
                    output_idx, output_labels = self.select_random_negative_candidates(input_idx, label_input, n=20)
                    loss, scores, correct_count = self.get_loss_combined_of_parent_child_and_n_neg(output_idx, output_labels, graph_candidates, 
                            sample_id, context_len)
                    return loss, scores, correct_count
                elif self.epoch > 0:
                    with torch.no_grad(): 
                        scores = self.score_candidate(input_idx, context_len)
                    output_idx, output_labels = self.select_crosenc_after_label_candidates(scores, input_idx, label_input, n=20)
                    loss, scores, correct_count = self.get_loss_combined_of_parent_child_and_n_neg(output_idx, output_labels, graph_candidates, 
                            sample_id, context_len)
                    return loss, scores, correct_count
            
            elif self.params['cross_enc_negative_selection'] == 'bienc_20_neg_and_prnt_chld_as_neg':
                output_idx, output_labels = self.select_random_negative_candidates(input_idx, label_input, n=20)
                loss, scores, correct_count = self.get_loss_combined_of_parent_child_and_n_neg(output_idx, output_labels, graph_candidates, 
                            sample_id, context_len)
                return loss, scores, correct_count
             
            elif self.params['cross_enc_negative_selection'] == 'only_bienc_20_neg':
                output_idx, output_labels = self.select_random_negative_candidates(input_idx, label_input, n=20)
                scores = self.score_candidate(output_idx, context_len)
                loss = F.cross_entropy(scores, output_labels, reduction="mean")
                correct_count = self.count_correctly_predicted(scores, output_labels)
                return loss, scores, correct_count
            
            elif self.params['cross_enc_negative_selection'] == 'only_bienc_63_neg':
                scores = self.score_candidate(input_idx, context_len)
                loss = F.cross_entropy(scores, label_input, reduction="mean")
                correct_count = self.count_correctly_predicted(scores, label_input)
                return loss, scores, correct_count
            
            elif self.params['cross_enc_negative_selection'] == 'prnt_chld_as_pos':
                output_idx, output_labels = self.select_random_negative_candidates(input_idx, label_input, n=20)
                loss, scores = self.get_loss_after_using_parent_child_as_pos(output_idx, output_labels, graph_candidates, 
                            sample_id, context_len)
                del output_idx
                del output_labels
                torch.cuda.empty_cache()
                return loss, scores
        else:
            scores = self.score_candidate(input_idx, context_len)
            if label_input_gt is not None:
                new_scores, new_labels = self.get_loss_after_removing_gt_from_neg(scores, label_input, label_input_gt)
                loss = F.cross_entropy(new_scores, new_labels, reduction="mean")
                return loss, scores
            else:
                loss = F.cross_entropy(scores, label_input, reduction="mean")
                return loss, scores

                if ranking_loss_fn is not None:
                    loss = ranking_loss_fn(new_scores, new_labels)
                else:

                    loss = F.cross_entropy(new_scores, new_labels, reduction="mean") 

            
            return loss, scores
        

    def select_crosenc_after_label_candidates(self, batch_scores, batch_input_idx, 
                                            batch_label_input, n=20):
        input_idx_list = []
        label_list = []
        for i in range(batch_label_input.shape[0]):
            scores = batch_scores[i]
            label_input = batch_label_input[i].item()
            sorted_indices = torch.argsort(scores, descending=True)
            label_rank = (sorted_indices == label_input).nonzero(as_tuple=True)[0].item()
            end_idx = min(label_rank + 1 + n, len(sorted_indices))
            selected_indices = sorted_indices[label_rank:end_idx]
            label_position = 0
            input_idx = batch_input_idx[i]
            selected_cands = input_idx[selected_indices]
            perm = torch.randperm(len(selected_cands), device=selected_cands.device)
            selected_cands = torch.index_select(selected_cands, 0, perm) # Random shuffle
            new_label_position = (perm == label_position).nonzero(as_tuple=True)[0].item() # Find new position of label (originally at index 0 before shuffle)
            input_idx_list.append(selected_cands)
            label_list.append(new_label_position)

        new_labels = torch.tensor(label_list, device=self.device)
        return input_idx_list, new_labels
        
    def get_loss_of_input_idx_from_crsenc_ranked_neg_after_label(self, batch_scores, batch_input_idx, 
                                                     batch_label_input,context_len, n=20):
        input_idx_list = []
        label_list = []
        for i in range(batch_label_input.shape[0]):
            scores = batch_scores[i]
            label_input = batch_label_input[i].item()
            sorted_indices = torch.argsort(scores, descending=True)
            label_rank = (sorted_indices == label_input).nonzero(as_tuple=True)[0].item()
            end_idx = min(label_rank + 1 + n, len(sorted_indices))
            selected_indices = sorted_indices[label_rank:end_idx]
            input_idx = batch_input_idx[i]
            selected_cands = input_idx[selected_indices]
            input_idx_list.append(selected_cands)
            label_list.append(0)
        samples = torch.stack(input_idx_list)
        new_labels = torch.tensor(label_list, device=self.device)
        scores = self.score_candidate(samples, context_len)
        loss = F.cross_entropy(scores, new_labels, reduction="mean")
        return loss
    
    def get_loss_of_score_from_crsenc_ranked_neg_after_label(self, batch_scores, batch_label_input, n=20):
        losses = []
        score_list = []
        label_list = []
        for i in range(batch_label_input.shape[0]):
            scores = batch_scores[i]
            sorted_indices = torch.argsort(scores, descending=True)
            label_input = batch_label_input[i].item()
            label_score = scores[label_input:label_input+1]
            label_rank = (sorted_indices == label_input).nonzero(as_tuple=True)[0].item()
            end_idx = min(label_rank + 1 + n, len(sorted_indices))
            selected_indices = sorted_indices[label_rank:end_idx]
            selected_scores = scores[selected_indices]
            perm = torch.randperm(len(selected_scores))
            selected_scores = selected_scores[perm]
            new_label_input = (perm == 0).nonzero(as_tuple=True)[0].item()
            score_list.append(selected_scores)
            label_list.append(new_label_input)
        scores = torch.stack(score_list)
        labels = torch.tensor(label_list, device=self.device)
        losses = F.cross_entropy(scores, labels, reduction="mean")
        return losses
    

    def count_correctly_predicted(self, scores, labels):
        predicted_labels = torch.argmax(scores, dim=1)
        correct = (predicted_labels == labels)
        correct_count = correct.sum().item()
        return correct_count



    def select_random_negative_candidates(self, input_idx,label_input, n=20):
        Batchs, Cands, Embd = input_idx.shape
        device = input_idx.device
        dtype = torch.long
        output_idx = torch.zeros(Batchs, n + 1, Embd, device=device, dtype=input_idx.dtype)
        output_labels = torch.zeros(Batchs, dtype=dtype, device=device)
        for i in range(Batchs):
            label_idx = label_input[i].item()
            # Get negative indices (all except the positive)
            all_indices = torch.arange(Cands, device=device)
            negative_mask = all_indices != label_idx
            negative_indices = all_indices[negative_mask]
            # Randomly select n=20 negatives
            perm = torch.randperm(negative_indices.size(0), device=device)[:n]
            selected_negative_indices = negative_indices[perm]
            negative_samples = input_idx[i, selected_negative_indices, :]
            # Get positive sample
            positive_sample = input_idx[i, label_idx, :].unsqueeze(0)
            # Insert positive at random position
            random_pos = torch.randint(0, n + 1, (1,), device=device).item()
            # Split negatives and insert positive
            if random_pos == 0:
                output_idx[i] = torch.cat([positive_sample, negative_samples], dim=0)
            elif random_pos == n:
                output_idx[i] = torch.cat([negative_samples, positive_sample], dim=0)
            else:
                output_idx[i] = torch.cat([
                    negative_samples[:random_pos],
                    positive_sample,
                    negative_samples[random_pos:]
                ], dim=0)
            
            output_labels[i] = random_pos
        output_idx, output_labels
        return output_idx, output_labels

    def get_loss_combined_of_parent_child_and_n_neg(self, input_idx, input_labels,
            graph_candidates, sample_id, context_len):
        device = self.device
        connected_candidates = graph_candidates[0]
        connected_candidate_len = graph_candidates[1]
        connected_labels = graph_candidates[2]
        correct_count = 0
        losses = []
        scores = []
        for i in range(len(input_idx)):
            # Reconstruct original (remove padding)
            length = connected_candidate_len[i].item()
            conn_cands = connected_candidates[i][:length]  # Shape: [num_cand_i]
            conn_label = connected_labels[i].item()
            # see_candidates_and_labels(self.params, self.tokenizer, original_candidates, 
            #                             original_label.item(), sample_id[i])
            prnt_chld_candidates = torch.cat(
                    [conn_cands[:conn_label], conn_cands[conn_label + 1:]])
            sample_candidates = input_idx[i]
            sample_label = input_labels[i].item()
            current_cands, current_label, prch_added_count = self.insert_parent_child_as_neg_at_random_positions(sample_candidates, 
                                                                                             sample_label, prnt_chld_candidates)
            self.prch_added_as_neg.append(prch_added_count)
            candidates_batch = current_cands.unsqueeze(0)  # Shape: [1, num_cand_i]
            conn_scores = self.score_candidate(candidates_batch, context_len)
            conn_scores = conn_scores.squeeze(0) 
            sample_loss = F.cross_entropy(conn_scores, torch.tensor(current_label, device=device), reduction="mean")
            losses.append(sample_loss)
            scores.append(conn_scores)
            pred_index = torch.argmax(conn_scores).item()
            if pred_index == current_label:
                correct_count+=1
        losses = torch.stack(losses).mean()
        return losses, scores, correct_count
    
    def get_loss_after_using_parent_child_as_pos(self, input_idx, input_labels,
            graph_candidates, sample_id, context_len):
        
        device = input_idx.device
        connected_candidates = graph_candidates[0]
        connected_candidate_len = graph_candidates[1]
        connected_labels = graph_candidates[2]

        accumulated_candidates = []
        accumulated_labels = []
        all_losses = []
        all_scores = []
        MINI_BATCH_SIZE = input_idx.shape[0]


       
        for i in range(len(input_idx)):
            label_idx = input_labels[i].item()

            accumulated_candidates.append(input_idx[i])
            accumulated_labels.append(label_idx)

            current_num_candidates = input_idx[i].shape[0]
            # Step 1: Exclude true candidate (19 remain)
            current_candidates = torch.cat([
                input_idx[i][:label_idx],
                input_idx[i][label_idx + 1:]
            ], dim=0)  # [19, embd_dim]
            # Step 2: Get valid connected candidates (exclude true label)
            conn_len = connected_candidate_len[i].item()
            valid_conn = connected_candidates[i][:conn_len]
            conn_label = connected_labels[i].item()
            
            connected_to_insert = torch.cat([
                valid_conn[:conn_label],
                valid_conn[conn_label + 1:]
            ], dim=0)


            bienc_neg_list = input_idx[i].tolist()
            
            # Step 3: Insert each connected candidate at random position
            for j in range(connected_to_insert.shape[0]):
                candidate_to_insert = connected_to_insert[j].unsqueeze(0)
                candidate_to_insert_list = candidate_to_insert.squeeze(0).tolist()

                # if prch already as bienc neg, skip it
                if candidate_to_insert_list in bienc_neg_list:
                    continue

                insert_pos = torch.randint(0, current_num_candidates-1, (1,)).item()
                
                # Insert at random_pos
                # new_candidates = insert(current_candidates, candidate_to_insert, insert_pos)
                # Perform the insertion
                if insert_pos == 0:
                    # Insert at the very beginning
                    new_candidates = torch.cat([candidate_to_insert, current_candidates], dim=0)
                elif insert_pos == current_num_candidates:
                    # Insert at the very end
                    new_candidates = torch.cat([current_candidates, candidate_to_insert], dim=0)
                else:
                    # Insert in the middle
                    new_candidates = torch.cat([
                        current_candidates[:insert_pos],
                        candidate_to_insert,
                        current_candidates[insert_pos:]
                    ], dim=0)
                new_label = insert_pos
                accumulated_candidates.append(new_candidates)
                accumulated_labels.append(new_label)

                # Process when batch is full
                if len(accumulated_candidates) >= MINI_BATCH_SIZE:
                    batch_cands = torch.stack(accumulated_candidates)
                    batch_labels = torch.tensor(accumulated_labels, dtype=torch.long, device=device)
                    scores = self.score_candidate(batch_cands, context_len)
                    all_scores.append(scores)
                    loss = F.cross_entropy(scores, batch_labels)
                    all_losses.append(loss.detach())
                    loss.backward()  # Backward immediately!
                    # Clear
                    accumulated_candidates = []
                    accumulated_labels = []
                    torch.cuda.empty_cache()

        if len(accumulated_candidates) > 0:
            # print(f"Processing final {len(accumulated_candidates)} samples")
            batch_cands = torch.stack(accumulated_candidates)
            batch_labels = torch.tensor(accumulated_labels, dtype=torch.long, device=device)
            scores = self.score_candidate(batch_cands, context_len)
            all_scores.append(scores)
            loss = F.cross_entropy(scores, batch_labels)
            all_losses.append(loss.detach())
            loss.backward()
            
            del batch_cands, batch_labels, scores, loss
            torch.cuda.empty_cache()

        losses = torch.stack(all_losses).mean()
        return losses, all_scores
    
    def insert_parent_child_at_random_positions(self, sample_candidates, sample_label, prnt_chld_candidates):
        device = sample_candidates.device
        num_to_insert = prnt_chld_candidates.shape[0]

        pr_ch_inds = []
        pr_ch_candidates_as_pos = []
        # Insert each parent-child candidate one by one
        for i in range(num_to_insert):
            current_num_candidates = sample_candidates.shape[0]
            
            # Random position: 0 to current_num_candidates (inclusive)
            # This means we can insert at the beginning, end, or anywhere in between
            insert_pos = torch.randint(0, current_num_candidates + 1, (1,), device=device).item()
            
            # Get the candidate to insert
            candidate_to_insert = prnt_chld_candidates[i].unsqueeze(0)  # [1, embedding_dim]

            
            # remove label
            original_candidates = sample_candidates.clone()
            
            current_candidates = torch.cat(
                    [original_candidates[:sample_label], original_candidates[sample_label + 1:]])
            
            # Perform the insertion
            if insert_pos == 0:
                # Insert at the very beginning
                current_candidates = torch.cat([candidate_to_insert, current_candidates], dim=0)
            elif insert_pos == current_num_candidates:
                # Insert at the very end
                current_candidates = torch.cat([current_candidates, candidate_to_insert], dim=0)
            else:
                # Insert in the middle
                current_candidates = torch.cat([
                    current_candidates[:insert_pos],
                    candidate_to_insert,
                    current_candidates[insert_pos:]
                ], dim=0)

            pr_ch_inds.append(insert_pos)
            pr_ch_candidates_as_pos.append(current_candidates)
        return pr_ch_candidates_as_pos, pr_ch_inds
    
    def insert_parent_child_as_neg_at_random_positions(self, sample_candidates, sample_label, prnt_chld_candidates):
        device = sample_candidates.device
        num_to_insert = prnt_chld_candidates.shape[0]
        current_candidates = sample_candidates.clone()
        bienc_neg_list = current_candidates.tolist()
        current_label = sample_label
        prch_added = 0
        for i in range(num_to_insert):
            if prch_added>=43: # i made it 43 to ensure candidates are not > 64 (21+43=64)
                break

            current_num_candidates = current_candidates.shape[0]
            candidate_to_insert = prnt_chld_candidates[i].unsqueeze(0)  # [1, embedding_dim]

            candidate_to_insert_list = candidate_to_insert.squeeze(0).tolist()
            # if prch already as bienc neg, skip it
            if candidate_to_insert_list in bienc_neg_list:
                continue
            prch_added+=1


            insert_pos = torch.randint(0, current_num_candidates + 1, (1,), device=device).item()
            if insert_pos <= current_label:
                current_label += 1
            if insert_pos == 0:
                current_candidates = torch.cat([candidate_to_insert, current_candidates], dim=0)
            elif insert_pos == current_num_candidates:
                current_candidates = torch.cat([current_candidates, candidate_to_insert], dim=0)
            else:
                current_candidates = torch.cat([
                    current_candidates[:insert_pos],
                    candidate_to_insert,
                    current_candidates[insert_pos:]
                ], dim=0)

        return current_candidates, current_label, prch_added

    def get_parent_child_score_and_loss(self, graph_candidates, sample_id, 
            context_len, exclude_label_score=True):
        
        connected_candidates = graph_candidates[0]
        connected_candidate_len = graph_candidates[1]
        connected_labels = graph_candidates[2]
        losses = []
        scores = []
        for i in range(len(connected_candidates)):
            # Reconstruct original (remove padding)
            length = connected_candidate_len[i].item()
            original_candidates = connected_candidates[i][:length]  # Shape: [num_cand_i]
            original_label = connected_labels[i]
            # see_candidates_and_labels(self.params, self.tokenizer, original_candidates, 
            #                             original_label.item(), sample_id[i])
            
            # Add batch dimension for score_candidate
            candidates_batch = original_candidates.unsqueeze(0)  # Shape: [1, num_cand_i]
            conn_scores = self.score_candidate(candidates_batch, context_len)
            conn_scores = conn_scores.squeeze(0) 
            sample_loss = F.cross_entropy(conn_scores, original_label, reduction="mean")
            losses.append(sample_loss)

            if exclude_label_score:
                exclude_index = original_label.item()
                conn_scores = torch.cat(
                    [conn_scores[:exclude_index], conn_scores[exclude_index + 1:]])
            scores.append(conn_scores)

        prnt_cld_loss = torch.stack(losses).mean()
        return scores, prnt_cld_loss


    
    def select_ranged_neg_sample(self,scores, label_input, start_rank=4 , end_rank=23):
        assert scores.dim() == 2, "scores must be [batch, num_candidates]"
        assert label_input.dim() == 1 and label_input.size(0) == scores.size(0), "label_input must be [batch]"
        B, C = scores.shape
        device = scores.device
        dtype = torch.long
        new_scores_list = []
        new_labels_list = []
        all_idx = torch.arange(C, device=device, dtype=dtype)
        for i in range(B):
            true_idx = label_input[i].item()
            # All negatives (exclude true)
            neg_idx = all_idx[all_idx != true_idx]
            # Sort negatives by score (descending) to get their ranking
            neg_scores = scores[i][neg_idx]
            sorted_indices = torch.argsort(neg_scores, descending=True)
            selected_sorted_idx = sorted_indices[start_rank:end_rank + 1] 
            sampled_neg = neg_idx[selected_sorted_idx]
            # Combine with true (true is at the end)
            sel = torch.cat([sampled_neg, torch.tensor([true_idx], device=device, dtype=dtype)], dim=0)
            # New label index is the last position (no shuffle)
            new_label = len(sel) - 1
            # Pick logits for these indices
            reduced_logits = scores[i].index_select(0, sel)  # shape [21] (20 negatives + 1 true)
            new_scores_list.append(reduced_logits.unsqueeze(0))
            new_labels_list.append(new_label)

        new_scores = torch.cat(new_scores_list, dim=0)
        new_labels = torch.tensor(new_labels_list, device=device)
        return new_scores, new_labels

    

    def symmetric_cross_entropy(self, predictions, targets, alpha=0.1, beta=1.0, num_classes=None):
        """
        Symmetric Cross Entropy - combines forward and reverse CE for noise robustness
        """
        if num_classes is None:
            num_classes = predictions.size(-1)
        
        # Forward cross entropy (standard)
        ce = F.cross_entropy(predictions, targets, reduction='none')
        
        # Reverse cross entropy
        pred_probs = F.softmax(predictions, dim=-1)
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Add small epsilon to avoid log(0)
        pred_probs = torch.clamp(pred_probs, min=1e-7, max=1.0)
        rce = -torch.sum(pred_probs * torch.log(target_one_hot + 1e-7), dim=-1)
        
        # Combine both losses
        loss = alpha * ce + beta * rce
        return loss.mean()
    
    def get_loss_after_removing_gt_from_neg(self, scores, label_input, gt):
        assert scores.dim() == 2, "scores must be [batch, num_candidates]"
        assert label_input.dim() == 1 and label_input.size(0) == scores.size(0), "label_input must be [batch]"
        B, C = scores.shape
        device = scores.device
        dtype = torch.long
        new_scores_list = []
        new_labels_list = []
        all_idx = torch.arange(C, device=device, dtype=dtype)
        VERY_NEG = torch.finfo(scores.dtype).min if scores.dtype.is_floating_point else -1e9
        if not scores.dtype.is_floating_point:
            VERY_NEG = -1e9 
        for i in range(B):
            true_idx = label_input[i].item()
            gt_idx = gt[i].item()
            if gt_idx == true_idx or gt_idx==-1:
                # Keep exactly as-is
                new_scores_list.append(scores[i].unsqueeze(0))
                new_labels_list.append(true_idx)
                continue
            new_true = true_idx - 1 if gt_idx < true_idx else true_idx
            # exclude label
            left = all_idx[:gt_idx]
            right = all_idx[gt_idx+1:]
            sel = torch.cat([left, right, all_idx[gt_idx:gt_idx+1]], dim=0)  # length C
            row = scores[i].index_select(0, sel)
            row[-1] = VERY_NEG # Neutralize last position (the excluded column)
            new_score = row.unsqueeze(0)
            with open('new_label.txt', 'a') as f:
                f.write(
                    f'peseudo label: {true_idx}, GT : {gt_idx} --> new label : {new_true}\nnew_score\n{new_score.tolist()}\n{scores[i].tolist()}\nold_score\n'
                )

            new_scores_list.append(new_score)
            new_labels_list.append(new_true)
        new_scores = torch.cat(new_scores_list, dim=0)
        new_labels = torch.tensor(new_labels_list, device=device, dtype=dtype)
        return new_scores, new_labels


    def select_neg_sample(self,scores, label_input, num_negatives):
        assert scores.dim() == 2, "scores must be [batch, num_candidates]"
        assert label_input.dim() == 1 and label_input.size(0) == scores.size(0), "label_input must be [batch]"
        B, C = scores.shape
        device = scores.device
        dtype = torch.long

        new_scores_list = []
        new_labels_list = []

        all_idx = torch.arange(C, device=device, dtype=dtype)

        for i in range(B):
            true_idx = label_input[i].item()

            # All negatives (exclude true)
            neg_idx = all_idx[all_idx != true_idx]
            # Sample n negatives
            sampled_neg = neg_idx[torch.randperm(len(neg_idx))[:num_negatives]]
            # Combine with true & shuffle
            sel = torch.cat([sampled_neg, torch.tensor([true_idx], device=device, dtype=dtype)], dim=0)
            shuffle = torch.randperm(sel.numel(), device=device)
            sel = sel[shuffle]
            # New label index is where sel == true_idx
            new_label = (sel == true_idx).nonzero(as_tuple=False).squeeze(-1).item()
            # Pick logits for these indices
            reduced_logits = scores[i].index_select(0, sel)  # shape [k+1]

            new_scores_list.append(reduced_logits.unsqueeze(0))
            new_labels_list.append(new_label)

        new_scores = torch.cat(new_scores_list, dim=0)              
        new_labels = torch.tensor(new_labels_list, device=device)     
        return new_scores, new_labels


def to_bert_input(token_idx, null_idx, segment_pos):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    if segment_pos > 0:
        segment_idx[:, segment_pos:] = token_idx[:, segment_pos:] > 0

    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    # token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


class TripletLossWithHardNegatives(nn.Module):
    """
    Triplet loss: anchor-positive distance < anchor-negative distance + margin
    With hard negative mining for better ranking
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores, correct_idx):
        """
        Args:
            scores: [batch_size, num_candidates]
            correct_idx: [batch_size]
        """
        batch_size = scores.size(0)
        
        # Positive scores (correct entities)
        positive_scores = scores.gather(1, correct_idx.unsqueeze(1)).squeeze(1)
        
        # For each example, find the HARDEST negative (highest scoring wrong entity)
        # This is key for improving hit@1
        negative_scores = scores.clone()
        negative_scores.scatter_(1, correct_idx.unsqueeze(1), float('-inf'))
        hard_negative_scores, _ = negative_scores.max(dim=1)
        
        # Triplet loss: positive should score higher than hardest negative
        loss = F.relu(self.margin + hard_negative_scores - positive_scores)
        
        return loss.mean()

class SmallLossTrainer:
    """
    Small-loss trick: gradually select samples with smaller losses
    Assumes correctly labeled samples have smaller losses
    """
    def __init__(self, forget_rate=0.2, num_gradual=10, exponent=1):
        self.forget_rate = forget_rate
        self.num_gradual = num_gradual
        self.exponent = exponent
        self.epoch = 0
    
    def get_forget_rate(self):
        """Calculate forget rate schedule"""
        return self.forget_rate * min(self.epoch / self.num_gradual, 1) ** self.exponent
    
    def compute_loss(self, predictions, targets):
        # Compute per-sample loss
        losses = F.cross_entropy(predictions, targets, reduction='none')
        
        # Sort losses and select small-loss samples
        num_remember = int((1 - self.get_forget_rate()) * len(losses))
        _, indices = torch.sort(losses)
        
        # Only backprop through samples with smallest losses
        selected_indices = indices[:num_remember]
        
        loss = losses[selected_indices].mean()
        return loss
    
    def step_epoch(self):
        self.epoch += 1



class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}