from __future__ import annotations


def time_based_split(df, group_col, time_col, train_fraction):
    """Time-aware split per group to reduce leakage."""
    train_idx = []
    test_idx = []

    for _, group in df.groupby(group_col, sort=False):
        group = group.sort_values(time_col)
        if len(group) < 2:
            train_idx.extend(group.index)
            continue

        cutoff = int(len(group) * train_fraction)
        cutoff = max(1, min(cutoff, len(group) - 1))
        train_idx.extend(group.index[:cutoff])
        test_idx.extend(group.index[cutoff:])

    return train_idx, test_idx
