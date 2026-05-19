import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch

from attenMixer import AreaAttnModel
from config import Best_setting, Dataset_setting, Model_setting
from utils.dataset import AttMixerDataset
from utils.utils import get_logger


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize(records, k):
    valid_ranks = [r['rank'] for r in records if r['rank'] > 0]
    if not records:
        return {'count': 0, 'hr': 0.0, 'avg_rank': 0.0, 'avg_loss': 0.0}
    return {
        'count': len(records),
        'hr': sum(r['hit_at_k'] for r in records) / len(records),
        'avg_rank': sum(valid_ranks) / len(valid_ranks) if valid_ranks else 0.0,
        'avg_loss': sum(r['loss'] for r in records) / len(records),
    }


def write_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def dataset_batch(dataset, index):
    return dataset.get_slice([index])


def evaluate_one(model, dataset, index, k, split):
    model.eval()
    with torch.no_grad():
        data = dataset_batch(dataset, index)
        targets, scores, candidate_set = model._forward(data)
        records = model._build_session_records(targets, scores, candidate_set, k, split, index)
    return records[0]


def evaluate_range(model, dataset, start, end, k, split):
    records = []
    model.eval()
    with torch.no_grad():
        for index in range(start, end):
            data = dataset_batch(dataset, index)
            targets, scores, candidate_set = model._forward(data)
            record = model._build_session_records(targets, scores, candidate_set, k, split, index)[0]
            records.append(record)
    return records


def online_update_one(model, dataset, index, steps):
    return model.online_update(dataset_batch(dataset, index), steps=steps)


class EpisodicTransitionMemory:
    def __init__(self, n_items, last_weight=1.0, item_weight=0.25, max_events=None):
        self.n_items = n_items
        self.last_weight = last_weight
        self.item_weight = item_weight
        self.max_events = max_events
        self.last_to_target = {}
        self.item_to_target = {}
        self.events = []

    def _add_edge(self, table, source, target):
        if source <= 0 or target <= 0:
            return
        targets = table.setdefault(int(source), {})
        targets[int(target)] = targets.get(int(target), 0.0) + 1.0

    def _remove_edge(self, table, source, target):
        targets = table.get(int(source))
        if not targets:
            return
        target = int(target)
        targets[target] = targets.get(target, 0.0) - 1.0
        if targets[target] <= 0:
            targets.pop(target, None)
        if not targets:
            table.pop(int(source), None)

    def add(self, session_items, target):
        clean_items = [int(item) for item in session_items if int(item) > 0]
        target = int(target)
        if not clean_items or target <= 0:
            return
        event = (clean_items, target)
        self.events.append(event)
        self._add_edge(self.last_to_target, clean_items[-1], target)
        for item in set(clean_items):
            self._add_edge(self.item_to_target, item, target)

        if self.max_events is not None and len(self.events) > self.max_events:
            old_items, old_target = self.events.pop(0)
            self._remove_edge(self.last_to_target, old_items[-1], old_target)
            for item in set(old_items):
                self._remove_edge(self.item_to_target, item, old_target)

    def candidate_scores(self, session_items, candidates):
        clean_items = [int(item) for item in session_items if int(item) > 0]
        candidates = [int(item) for item in candidates if int(item) > 0]
        if not clean_items or not candidates:
            return {}

        scores = {candidate: 0.0 for candidate in candidates}
        last_edges = self.last_to_target.get(clean_items[-1], {})
        for candidate in candidates:
            scores[candidate] += self.last_weight * last_edges.get(candidate, 0.0)

        unique_items = set(clean_items)
        for item in unique_items:
            item_edges = self.item_to_target.get(item, {})
            for candidate in candidates:
                scores[candidate] += self.item_weight * item_edges.get(candidate, 0.0)

        max_score = max(scores.values()) if scores else 0.0
        if max_score > 0:
            scores = {candidate: score / max_score for candidate, score in scores.items()}
        return scores

    def __len__(self):
        return len(self.events)


def session_items(dataset, index):
    return list(dataset.data[0][index])


def session_target(dataset, index):
    target = dataset.data[1][index]
    if isinstance(target, (list, tuple, np.ndarray)):
        return int(np.asarray(target).reshape(-1)[0])
    return int(target)


def apply_episodic_scores(scores, candidate_set, memory_scores, beta):
    if not memory_scores or beta <= 0:
        return scores
    adjusted = scores.clone()
    for item, score in memory_scores.items():
        if 1 <= item <= adjusted.size(1):
            adjusted[:, item - 1] = adjusted[:, item - 1] + beta * float(score)
    return adjusted


def evaluate_one_with_memory(model, dataset, index, k, split, memory=None, memory_beta=0.0):
    model.eval()
    with torch.no_grad():
        data = dataset_batch(dataset, index)
        targets, scores, candidate_set = model._forward(data)
        memory_scores = {}
        if memory is not None:
            candidates = candidate_set.detach().cpu().view(-1).tolist() if candidate_set.numel() > 0 else []
            memory_scores = memory.candidate_scores(session_items(dataset, index), candidates)
            scores = apply_episodic_scores(scores, candidate_set, memory_scores, memory_beta)
        records = model._build_session_records(targets, scores, candidate_set, k, split, index)
    record = records[0]
    if memory is not None:
        record['episodic_memory_size'] = len(memory)
        record['memory_beta'] = memory_beta
        record['memory_nonzero_candidates'] = sum(1 for value in memory_scores.values() if value > 0)
    return record


def evaluate_range_with_memory(model, dataset, start, end, k, split, memory=None, memory_beta=0.0):
    return [
        evaluate_one_with_memory(model, dataset, index, k, split, memory, memory_beta)
        for index in range(start, end)
    ]


def rolling_mean(values, window):
    rolled = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start:index + 1]
        rolled.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return rolled


def record_train_session_changes(model, dataset, max_sessions, k, online_steps):
    records = []
    limit = min(max_sessions, len(dataset))
    for index in range(limit):
        before = evaluate_one(model, dataset, index, k, 'train_before_update')
        update_loss = online_update_one(model, dataset, index, online_steps)
        after = evaluate_one(model, dataset, index, k, 'train_after_update')
        records.append({
            'session_index': index,
            'target': before['target'],
            'pre_rank': before['rank'],
            'post_rank': after['rank'],
            'rank_delta': before['rank'] - after['rank'],
            f'pre_hit_at_{k}': before['hit_at_k'],
            f'post_hit_at_{k}': after['hit_at_k'],
            'pre_loss': before['loss'],
            'post_loss': after['loss'],
            'loss_delta': before['loss'] - after['loss'],
            'online_update_loss': update_loss,
            'online_steps': online_steps,
        })
    return records


def block_continual_test_with_snapshots(
    model, dataset, max_sessions, k, online_steps, update_every,
    memory_beta=1.0, memory_last_weight=1.0, memory_item_weight=0.25,
    memory_max_events=None, rolling_window=20,
):
    session_records = []
    snapshot_records = []
    limit = min(max_sessions, len(dataset))
    pending_feedback = []
    updates_seen = 0
    memory = EpisodicTransitionMemory(
        model.model.n_node - 1,
        last_weight=memory_last_weight,
        item_weight=memory_item_weight,
        max_events=memory_max_events,
    )

    for index in range(limit):
        session_record = evaluate_one_with_memory(
            model, dataset, index, k, 'test_elm_block_continual', memory, memory_beta
        )
        session_record['update_applied_after_this_session'] = 0
        session_record['update_every'] = update_every
        session_records.append(session_record)

        target = session_target(dataset, index)
        memory.add(session_items(dataset, index), target)
        pending_feedback.append(index)

        should_update = len(pending_feedback) >= update_every or index == limit - 1
        model_updated_at_t = 0
        block_update_avg_loss = ''
        if should_update:
            update_losses = []
            for feedback_index in pending_feedback:
                update_losses.append(online_update_one(model, dataset, feedback_index, online_steps))
            pending_feedback = []
            updates_seen += 1
            model_updated_at_t = 1
            block_update_avg_loss = sum(update_losses) / len(update_losses)
            session_records[-1]['update_applied_after_this_session'] = 1
            session_records[-1]['block_update_avg_loss'] = block_update_avg_loss

        future_records = evaluate_range_with_memory(
            model, dataset, index + 1, limit, k, 'future_eval_frozen_elm', memory, memory_beta
        )
        future_summary = summarize(future_records, k)
        snapshot_records.append({
            'time_t': index + 1,
            'last_session_index': index,
            'last_target': session_record['target'],
            'last_rank': session_record['rank'],
            f'last_hit_at_{k}': session_record['hit_at_k'],
            'last_loss': session_record['loss'],
            'future_session_count': future_summary['count'],
            f'future_avg_hit_at_{k}': future_summary['hr'],
            'future_avg_rank': future_summary['avg_rank'],
            'future_avg_loss': future_summary['avg_loss'],
            'episodic_memory_size': len(memory),
            'updates_seen_so_far': updates_seen,
            'model_updated_at_t': model_updated_at_t,
            'block_update_avg_loss': block_update_avg_loss,
        })

    cumulative_records = []
    observed_hits = [record['hit_at_k'] for record in session_records]
    observed_rolling = rolling_mean(observed_hits, rolling_window)
    future_hits = [record[f'future_avg_hit_at_{k}'] for record in snapshot_records]
    future_rolling = rolling_mean(future_hits, rolling_window)

    for index in range(limit):
        prefix = session_records[:index + 1]
        prefix_summary = summarize(prefix, k)
        cumulative_records.append({
            'time_t': index + 1,
            'observed_session_count': prefix_summary['count'],
            f'observed_avg_hit_at_{k}': prefix_summary['hr'],
            f'observed_rolling{rolling_window}_hit_at_{k}': observed_rolling[index],
            'observed_avg_rank': prefix_summary['avg_rank'],
            'observed_avg_loss': prefix_summary['avg_loss'],
            f'future_rolling{rolling_window}_avg_hit_at_{k}': future_rolling[index],
        })

    return session_records, snapshot_records, cumulative_records

def plot_hit_at_k(cumulative_records, snapshot_records, k, output_path, rolling_window=20):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    times = [r['time_t'] for r in cumulative_records]
    observed_key = f'observed_rolling{rolling_window}_hit_at_{k}'
    future_key = f'future_rolling{rolling_window}_avg_hit_at_{k}'
    observed = [r.get(observed_key, r[f'observed_avg_hit_at_{k}']) for r in cumulative_records]
    future = [r.get(future_key, snapshot_records[i][f'future_avg_hit_at_{k}']) for i, r in enumerate(cumulative_records)]

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, observed, marker='o', label=f'Observed rolling-{rolling_window} HIT@{k}')
    plt.plot(times, future, marker='s', label=f'Frozen future rolling-{rolling_window} HIT@{k}')
    plt.xlabel('Time t / processed test sessions')
    plt.ylabel(f'HIT@{k}')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_compare_ready_hit_at_k(cumulative_records, k, output_path, rolling_window=20, update_every=5):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    times = [r['time_t'] for r in cumulative_records]
    key = f'observed_rolling{rolling_window}_hit_at_{k}'
    y = [r.get(key, r[f'observed_avg_hit_at_{k}']) for r in cumulative_records]
    flush_x = [time for time in times if time % update_every == 0]
    flush_y = [y[time - 1] for time in flush_x if time - 1 < len(y)]

    plt.figure(figsize=(13.2, 6.05))
    plt.plot(times, y, color='#1f77b4', linewidth=2.0, label=f'AttenMixer-ELM-Block{update_every}')
    plt.scatter(flush_x, flush_y, color='#ff7f0e', s=26, zorder=3, label=f'gradient update every {update_every} sessions')
    plt.title(f'AttenMixer test mean HIT@{k} rolling (window={rolling_window}), cadence={update_every}')
    plt.xlabel('Snapshot index')
    plt.ylabel(f'HIT@{k}')
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def load_data(
    data_dir, sample_num, seed, max_raw_sessions,
    max_train_raw_sessions=None, max_test_sessions=None, trim_tail_sessions=0,
):
    train_data = np.load(data_dir / f'train_sample_{sample_num}.npy', allow_pickle=True).tolist()
    test_data = np.load(data_dir / 'test.npy', allow_pickle=True).tolist()
    candidate_data = np.load(data_dir / f'test_candidate_{seed}.npy', allow_pickle=True).tolist()

    if max_raw_sessions is not None:
        train_data = train_data[:max_raw_sessions]
        test_data = test_data[:max_raw_sessions]
        candidate_data = candidate_data[:max_raw_sessions]

    if max_train_raw_sessions is not None:
        train_data = train_data[:max_train_raw_sessions]

    if trim_tail_sessions > 0:
        test_data = test_data[:-trim_tail_sessions]
        candidate_data = candidate_data[:-trim_tail_sessions]

    if max_test_sessions is not None:
        test_data = test_data[:max_test_sessions]
        candidate_data = candidate_data[:max_test_sessions]

    return train_data, test_data, candidate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='bundle', help='bundle/games/ml-1m')
    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max_raw_sessions', type=int, default=None)
    parser.add_argument('--max_train_raw_sessions', type=int, default=None)
    parser.add_argument('--max_test_sessions', type=int, default=None)
    parser.add_argument('--trim_tail_sessions', type=int, default=0)
    parser.add_argument('--max_record_sessions', type=int, default=None)
    parser.add_argument('--online_steps', type=int, default=1)
    parser.add_argument('--update_every', type=int, default=5)
    parser.add_argument('--memory_beta', type=float, default=1.0)
    parser.add_argument('--memory_last_weight', type=float, default=1.0)
    parser.add_argument('--memory_item_weight', type=float, default=0.25)
    parser.add_argument('--memory_max_events', type=int, default=None)
    parser.add_argument('--rolling_window', type=int, default=20)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    init_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else root / 'Dataset' / 'ID'
    output_dir = Path(args.output_dir) if args.output_dir else root / 'res' / 'continual_debug' / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data, candidate_data = load_data(
        data_dir, args.sample_num, args.seed, args.max_raw_sessions,
        max_train_raw_sessions=args.max_train_raw_sessions,
        max_test_sessions=args.max_test_sessions,
        trim_tail_sessions=args.trim_tail_sessions,
    )
    if args.max_record_sessions is None:
        args.max_record_sessions = len(test_data)

    model_config = {**Model_setting['AttenMixer'], **Best_setting['AttenMixer'][args.dataset]}
    model_config['gpu'] = args.gpu
    model_config['epochs'] = args.epochs
    model_config['batch_size'] = args.batch_size

    record_config = {**model_config, 'batch_size': 1}
    logger = get_logger(f'continual_attenmixer_{args.dataset}_debug')

    train_dataset = AttMixerDataset(train_data, model_config, isTrain=True)
    train_loader = train_dataset.get_loader(model_config, shuffle=False)
    train_record_loader = train_dataset.get_loader(record_config, shuffle=False)

    test_dataset = AttMixerDataset(
        test_data, model_config, candidate_set=candidate_data, isTrain=False
    )
    test_loader = test_dataset.get_loader(record_config, shuffle=False)

    num_node = Dataset_setting[args.dataset]['num_node'] + 1
    model = AreaAttnModel(model_config, num_node, logger=logger)

    logger.info(
        f'Continual debug: train_raw={len(train_data)}, test_raw={len(test_data)}, '
        f'train_examples={len(train_dataset)}, test_examples={len(test_dataset)}, '
        f'max_record_sessions={args.max_record_sessions}'
    )

    model.fit(train_loader)
    fitted_state = copy.deepcopy(model.state_dict())

    train_change_path = output_dir / f'train_session_changes_hit@{args.k}.csv'
    test_session_path = output_dir / f'test_elm_block_continual_sessions_hit@{args.k}.csv'
    test_snapshot_path = output_dir / f'test_elm_time_t_frozen_future_avg_hit@{args.k}.csv'
    test_cumulative_path = output_dir / f'test_elm_cumulative_observed_avg_hit@{args.k}.csv'
    plot_path = output_dir / f'elm_hit@{args.k}_rolling{args.rolling_window}_over_time.png'
    compare_ready_plot_path = output_dir / f'attmixer_elm_hit@{args.k}_rolling{args.rolling_window}_compare_ready.png'

    train_change_records = record_train_session_changes(
        model, train_dataset, args.max_record_sessions, args.k, args.online_steps
    )
    model.load_state_dict(fitted_state)
    test_records, snapshot_records, cumulative_records = block_continual_test_with_snapshots(
        model, test_dataset, args.max_record_sessions, args.k, args.online_steps, args.update_every,
        memory_beta=args.memory_beta,
        memory_last_weight=args.memory_last_weight,
        memory_item_weight=args.memory_item_weight,
        memory_max_events=args.memory_max_events,
        rolling_window=args.rolling_window,
    )

    write_csv(train_change_path, train_change_records)
    write_csv(test_session_path, test_records)
    write_csv(test_snapshot_path, snapshot_records)
    write_csv(test_cumulative_path, cumulative_records)
    plotted = plot_hit_at_k(cumulative_records, snapshot_records, args.k, plot_path, args.rolling_window)
    compare_plotted = plot_compare_ready_hit_at_k(
        cumulative_records, args.k, compare_ready_plot_path, args.rolling_window, args.update_every
    )

    train_summary = summarize([
        {'rank': r['post_rank'], 'hit_at_k': r[f'post_hit_at_{args.k}'], 'loss': r['post_loss']}
        for r in train_change_records
    ], args.k)
    test_summary = summarize(test_records, args.k)
    logger.info(f'Train change records written to: {train_change_path}')
    logger.info(f'Test session records written to: {test_session_path}')
    logger.info(f'Test frozen-future averages written to: {test_snapshot_path}')
    logger.info(f'Test cumulative observed averages written to: {test_cumulative_path}')
    logger.info(f'HIT@{args.k} plot written to: {plot_path if plotted else "not created; matplotlib is missing"}')
    logger.info(f'Compare-ready HIT@{args.k} plot written to: {compare_ready_plot_path if compare_plotted else "not created; matplotlib is missing"}')
    logger.info(f'Train post-update summary@{args.k}: {train_summary}')
    logger.info(f'Test ELM block-continual observed summary@{args.k}: {test_summary}')


if __name__ == '__main__':
    main()
