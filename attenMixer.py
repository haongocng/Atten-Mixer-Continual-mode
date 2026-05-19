import csv
import math
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


# requisite class
class LastAttenion(Module):

    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_lp_pool=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_lp_pool = use_lp_pool
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):

        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1),
                                          self.hidden_size // self.heads)  # batch_size x seq_length x latent_size
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        assert not torch.isnan(q0).any()
        assert not torch.isnan(q1).any()
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        assert not torch.isnan(alpha).any()
        if self.use_lp_pool == True:
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.last_layernorm(a)
        return a, alpha


class SessionGraphAttn(Module):
    def __init__(self, opt, n_node):
        super(SessionGraphAttn, self).__init__()
        self.hidden_size = opt['item_embedding_dim']
        self.n_node = n_node
        self.norm = opt['norm']
        self.scale = opt['scale']
        self.batch_size = opt['batch_size']
        self.heads = opt['heads']
        self.use_lp_pool = opt['use_lp_pool']
        self.softmax = opt['softmax']
        self.dropout = opt['dropout']
        self.last_k = opt['last_k']
        self.dot = opt['dot']
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.l_p = opt['l_p']
        self.mattn = LastAttenion(self.hidden_size, self.heads, self.dot, self.l_p, last_k=self.last_k,
                                  use_lp_pool=self.use_lp_pool)
        self.linear_q = nn.ModuleList()
        for i in range(self.last_k):
            self.linear_q.append(nn.Linear((i + 1) * self.hidden_size, self.hidden_size))

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get(self, i, hidden, alias_inputs):
        return hidden[i][alias_inputs[i]]

    def compute_scores(self, hidden, mask):

        hts = []

        lengths = torch.sum(mask, dim=1)
        batch_index = torch.arange(mask.size(0), device=hidden.device).long()

        for i in range(self.last_k):
            hts.append(self.linear_q[i](torch.cat(
                [hidden[batch_index, torch.clamp(lengths - (j + 1), -1, 1000).long()] for j in
                 range(i + 1)], dim=-1)).unsqueeze(1))

        ht0 = hidden[batch_index, (torch.sum(mask, 1) - 1).long()]

        hts = torch.cat(hts, dim=1)
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)

        hidden1 = hidden
        hidden = hidden1[:, :mask.size(1)]

        ais, weights = self.mattn(hts, hidden, mask)
        ais = ais.reshape(hidden.size(0), -1)
        a = self.linear_transform(torch.cat((ais, ht0), 1))

        b = self.embedding.weight[1:]

        if self.norm:
            a = a.div(torch.norm(a, p=2, dim=1, keepdim=True) + 1e-12)
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
        b = F.dropout(b, self.dropout, training=self.training)
        scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 16 * scores
        return scores

    def forward(self, inputs):

        hidden = self.embedding(inputs)

        if self.norm:
            hidden = hidden.div(torch.norm(hidden, p=2, dim=-1, keepdim=True) + 1e-12)

        hidden = F.dropout(hidden, self.dropout, training=self.training)

        return hidden
        
# requisite end

class AreaAttnModel(Module):

    def __init__(self, opt, n_node, logger=None):
        super().__init__()

        self.opt = opt
        self.cnt = 0
        self.best_res = [0, 0]
        self.model = SessionGraphAttn(opt, n_node)
        self.loss = nn.Parameter(torch.Tensor(1))
        self.batch_size = opt['batch_size']
        self.loss_function = self.model.loss_function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['learning_rate'], weight_decay=self.opt['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt['lr_dc_step'], gamma=self.opt['lr_dc'])
        self.logger = logger
        gpuid = opt['gpu']
        self.devices = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    def forward(self, *args):

        return self.model(*args)

    
    def _remove_loader_axis(self, tensor):
        if torch.is_tensor(tensor) and tensor.dim() > 0 and tensor.size(0) == 1:
            return tensor.squeeze(0)
        return tensor

    def _ensure_batch_axis(self, alias_inputs, A, items, mask, mask1, targets, n_node, candidate_set):
        if alias_inputs.dim() == 1:
            alias_inputs = alias_inputs.unsqueeze(0)
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if items.dim() == 1:
            items = items.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask1.dim() == 1:
            mask1 = mask1.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        if n_node.dim() == 0:
            n_node = n_node.unsqueeze(0)
        if candidate_set.numel() > 0 and candidate_set.dim() == 1:
            candidate_set = candidate_set.unsqueeze(0)
        return alias_inputs, A, items, mask, mask1, targets, n_node, candidate_set

    def _forward(self, data):
        alias_inputs, A, items, mask, mask1, targets, n_node, candidate_set = [
            self._remove_loader_axis(x) for x in data
        ]
        alias_inputs, A, items, mask, mask1, targets, n_node, candidate_set = self._ensure_batch_axis(
            alias_inputs, A, items, mask, mask1, targets, n_node, candidate_set
        )
        alias_inputs = alias_inputs.long().to(self.devices)
        A = A.float().to(self.devices)
        items = items.long().to(self.devices)
        mask = mask.long().to(self.devices)
        mask1 = mask1.long().to(self.devices)
        targets = targets.view(-1).long()
        n_node = n_node.view(-1).long().to(self.devices)
        candidate_set = candidate_set.long().to(self.devices)

        hidden = self(items)

        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        max_n_node = int(torch.max(n_node).item())
        seq_hidden = torch.cat((seq_hidden, hidden[:, max_n_node:]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)

        if self.opt['norm']:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt['item_embedding_dim'])
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        return targets, scores, candidate_set

    def _build_session_records(self, targets, scores, candidate_set, k, split, start_index=0):
        targets_cpu = targets.detach().cpu().view(-1)
        scores_cpu = scores.detach().cpu()
        candidate_cpu = candidate_set.detach().cpu()
        losses = F.cross_entropy(scores, targets.to(self.devices) - 1, reduction='none').detach().cpu()
        records = []

        for row in range(scores_cpu.size(0)):
            target = int(targets_cpu[row].item())
            if candidate_cpu.numel() > 0:
                candidates = candidate_cpu[row].view(-1).long()
                cand_scores = scores_cpu[row].gather(0, candidates - 1)
                target_positions = torch.nonzero(candidates == target, as_tuple=False).view(-1)
                if len(target_positions) > 0:
                    target_score = cand_scores[target_positions[0]]
                    rank = int((cand_scores > target_score).sum().item() + 1)
                else:
                    target_score = torch.tensor(float('nan'))
                    rank = -1
                top_count = min(k, cand_scores.numel())
                top_indices = torch.topk(cand_scores, top_count).indices
                top_items = candidates.gather(0, top_indices).tolist()
                candidate_count = int(candidates.numel())
            else:
                target_score = scores_cpu[row, target - 1]
                rank = int((scores_cpu[row] > target_score).sum().item() + 1)
                top_count = min(k, scores_cpu.size(1))
                top_items = (torch.topk(scores_cpu[row], top_count).indices + 1).tolist()
                candidate_count = int(scores_cpu.size(1))

            records.append({
                'split': split,
                'session_index': start_index + row,
                'target': target,
                'rank': rank,
                'hit_at_k': int(0 < rank <= k),
                'loss': float(losses[row].item()),
                'target_score': float(target_score.item()),
                'top1': int(top_items[0]) if top_items else -1,
                'topk': ' '.join(str(item) for item in top_items),
                'candidate_count': candidate_count,
            })
        return records

    def _write_session_records(self, record_path, records, append=True):
        if record_path is None or not records:
            return
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        write_header = not append or not os.path.exists(record_path)
        with open(record_path, 'a' if append else 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(records)

    def predict_with_records(self, data_loader, k=10, split='test', record_path=None, max_sessions=None, append=False):
        self.eval()
        all_records = []
        seen = 0
        for data in data_loader:
            with torch.no_grad():
                targets, scores, candidate_set = self._forward(data)
                records = self._build_session_records(targets, scores, candidate_set, k, split, seen)
            if max_sessions is not None:
                records = records[:max_sessions - seen]
            self._write_session_records(record_path, records, append=(append or seen > 0))
            all_records.extend(records)
            seen += len(records)
            if max_sessions is not None and seen >= max_sessions:
                break
        return all_records

    def online_update(self, data, steps=1):
        self.train()
        last_loss = 0.0
        for _ in range(steps):
            self.optimizer.zero_grad()
            targets, scores, _ = self._forward(data)
            targets = targets.long().to(self.devices)
            loss = self.model.loss_function(scores, targets - 1)
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())
        return last_loss

    def continual_predict(self, data_loader, k=10, record_path=None, max_sessions=None, online_steps=1):
        all_records = []
        seen = 0
        for data in data_loader:
            self.eval()
            with torch.no_grad():
                targets, scores, candidate_set = self._forward(data)
                records = self._build_session_records(targets, scores, candidate_set, k, 'test_continual', seen)
            if max_sessions is not None:
                records = records[:max_sessions - seen]

            update_loss = self.online_update(data, steps=online_steps)
            for record in records:
                record['online_update_loss'] = update_loss
                record['online_steps'] = online_steps

            self._write_session_records(record_path, records, append=(seen > 0))
            all_records.extend(records)
            seen += len(records)
            if max_sessions is not None and seen >= max_sessions:
                break
        return all_records


    def fit(self, train_data, validation_data=None):
        gpuid = int(self.devices.index)
        self.cuda(gpuid) if torch.cuda.is_available() else self.cpu()
        # self.cuda(1) if torch.cuda.is_available() else self.cpu()
        if self.logger:
            self.logger.info('Start training...')

        last_loss = 0.
        for epoch in range(1, self.opt['epochs'] + 1):
            self.train()
            total_loss = []
            current_loss = 0.
            train_loader = train_data
            for data in tqdm(train_loader):
                self.optimizer.zero_grad()
                targets, scores, _ = self._forward(data)
                targets = targets.long().to(self.devices)
                loss = self.model.loss_function(scores, targets - 1)
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
                # return loss
            self.scheduler.step()
            current_loss = np.mean(total_loss)
            delta_loss = current_loss - last_loss
            if abs(delta_loss) < 1e-5:
                if self.logger:
                    self.logger.info(f'Early stop at epoch {epoch}')
                break
            last_loss = current_loss
            
            s = ''
            if validation_data:
                valid_loss = self.evaluate(validation_data)
                s = f'\tValidation Loss: {valid_loss:.4f}'
            if self.logger:
                self.logger.info(f'Training Epoch: {epoch}\tLoss: {np.mean(total_loss):.4f}' + s)
    
    def evaluate(self, validation_data):
        self.eval()
        valid_loss = []
        valid_loader = torch.utils.data.DataLoader(validation_data, num_workers=4, batch_size=self.batch_size,
                                                   shuffle=False, pin_memory=True)
        for data in tqdm(valid_loader):
            targets, scores = self._forward(data)
            targets = targets.long().to(self.devices)
            loss = self.loss_function(scores, (targets - 1).squeeze())
            valid_loss.append(loss.item())
        return np.mean(valid_loss)

    def predict(self, test_data, k=15):
        if self.logger:
            self.logger.info('Start predicting...')
        self.eval()
        test_loader = test_data
        preds, last_item = torch.tensor([]), torch.tensor([])
        for data in test_loader:
            targets, scores, candidate_data = self._forward(data)
            sub_scores = torch.gather(scores, 1, candidate_data-1).topk(k)[1]
            sub_scores = torch.gather(candidate_data, 1, sub_scores)
            preds = torch.cat((preds, sub_scores.cpu()), 0)
            last_item = torch.cat((last_item, torch.tensor(targets)), 0)
        return preds, last_item