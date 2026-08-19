#!/usr/bin/env python3
"""Generate version comparison plots from benchmark data."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Version data from benchmarks
versions = {
    'v2 (BM25+TF-IDF)': {
        'hit_at_5': 0.900,
        'top1': 0.633,
        'mrr': 0.736,
        'latency': 2.8,
        'keyword_mrr': 0.180,
        'typo_top1': 0.90,
    },
    'v3 (BM25+Dense)': {
        'hit_at_5': 0.892,
        'top1': 0.725,
        'mrr': 0.787,
        'latency': 1.9,
        'keyword_mrr': 0.180,
        'typo_top1': 0.90,
    },
    'v4 (BM25+ngram+Dense+Reranker)': {
        'hit_at_5': 0.900,
        'top1': 0.650,
        'mrr': 0.775,
        'latency': 15.8,
        'keyword_mrr': 0.333,
        'typo_top1': 1.00,
    }
}

# Per-format data (from v3 and v4 benchmarks)
format_data = {
    'v3': {
        'verbatim':     {'hit': 1.00, 'top1': 0.90, 'mrr': 0.942, 'lat': 9820},
        'paraphrase':   {'hit': 1.00, 'top1': 0.95, 'mrr': 0.975, 'lat': 279},
        'typo':         {'hit': 1.00, 'top1': 0.90, 'mrr': 0.942, 'lat': 257},
        'conversational': {'hit': 1.00, 'top1': 0.85, 'mrr': 0.917, 'lat': 277},
        'reworded':     {'hit': 0.95, 'top1': 0.65, 'mrr': 0.760, 'lat': 265},
        'keyword_only': {'hit': 0.45, 'top1': 0.10, 'mrr': 0.180, 'lat': 243},
    },
    'v4': {
        'verbatim':     {'hit': 1.00, 'top1': 0.75, 'mrr': 0.875, 'lat': 50359},
        'paraphrase':   {'hit': 1.00, 'top1': 0.50, 'mrr': 0.750, 'lat': 7000},
        'typo':         {'hit': 1.00, 'top1': 1.00, 'mrr': 1.000, 'lat': 7500},
        'conversational': {'hit': 1.00, 'top1': 0.67, 'mrr': 0.833, 'lat': 7000},
        'reworded':     {'hit': 1.00, 'top1': 0.67, 'mrr': 0.833, 'lat': 8000},
        'keyword_only': {'hit': 0.33, 'top1': 0.33, 'mrr': 0.333, 'lat': 6200},
    }
}

output_dir = Path('data/plots')
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
colors = ['#3b82f6', '#10b981', '#f59e0b']
format_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

# ============================================================
# Plot 1: Overall Metrics Comparison (Grouped Bar Chart)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Retrieval Pipeline Version Comparison: v2 → v3 → v4', fontsize=16, fontweight='bold')

metrics = ['hit_at_5', 'top1', 'mrr', 'latency']
metric_labels = ['Hit@5', 'Top-1 Accuracy', 'MRR', 'Latency (s)']
x = np.arange(len(versions))
width = 0.2

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    values = [versions[v][metric] for v in versions]
    bars = ax.bar(x, values, width, label=list(versions.keys()), color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(versions.keys()), rotation=15, ha='right')
    if metric == 'latency':
        ax.set_ylim(0, max([versions[v]['latency'] for v in versions]) * 1.2)
        ax.set_ylabel('Seconds')
    else:
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
    for bar, val in zip(bars, values):
        if metric == 'latency':
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}s', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'version_comparison_overall.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 2: Per-Format Comparison (v3 vs v4)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Per-Format Comparison: v3 vs v4', fontsize=16, fontweight='bold')

formats = ['verbatim', 'paraphrase', 'typo', 'conversational', 'reworded', 'keyword_only']
format_labels = ['Verbatim', 'Paraphrase', 'Typo', 'Conversational', 'Reworded', 'Keyword-only']

for idx, (fmt, label) in enumerate(zip(formats, format_labels)):
    ax = axes[idx // 3, idx % 3]
    
    v3_data = format_data['v3'][fmt]
    v4_data = format_data['v4'][fmt]
    
    x_pos = np.arange(3)
    width = 0.35
    
    v3_vals = [v3_data['hit'], v3_data['top1'], v3_data['mrr']]
    v4_vals = [v4_data['hit'], v4_data['top1'], v4_data['mrr']]
    
    bars1 = ax.bar(x_pos - width/2, v3_vals, width, label='v3', color='#3b82f6', edgecolor='white')
    bars2 = ax.bar(x_pos + width/2, v4_vals, width, label='v4', color='#f59e0b', edgecolor='white')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Hit@5', 'Top-1', 'MRR'])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    
    for bar, val in zip(bars1, v3_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, v4_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'version_comparison_per_format.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 3: Radar Chart - Capabilities
# ============================================================
categories = ['Hit@5', 'Top-1', 'MRR', 'Typo Robustness', 'Keyword Handling', 'Latency (inv)']
N = len(categories)

# Normalize latency (lower is better, so invert)
v2_lat = 1 - min(1, versions['v2 (BM25+TF-IDF)']['latency'] / 20)
v3_lat = 1 - min(1, versions['v3 (BM25+Dense)']['latency'] / 20)
v4_lat = 1 - min(1, versions['v4 (BM25+ngram+Dense+Reranker)']['latency'] / 20)

v2_vals = [versions['v2 (BM25+TF-IDF)']['hit_at_5'], versions['v2 (BM25+TF-IDF)']['top1'], 
           versions['v2 (BM25+TF-IDF)']['mrr'], versions['v2 (BM25+TF-IDF)']['typo_top1'],
           versions['v2 (BM25+TF-IDF)']['keyword_mrr'], v2_lat]
v3_vals = [versions['v3 (BM25+Dense)']['hit_at_5'], versions['v3 (BM25+Dense)']['top1'],
           versions['v3 (BM25+Dense)']['mrr'], versions['v3 (BM25+Dense)']['typo_top1'],
           versions['v3 (BM25+Dense)']['keyword_mrr'], v3_lat]
v4_vals = [versions['v4 (BM25+ngram+Dense+Reranker)']['hit_at_5'], 
           versions['v4 (BM25+ngram+Dense+Reranker)']['top1'],
           versions['v4 (BM25+ngram+Dense+Reranker)']['mrr'],
           versions['v4 (BM25+ngram+Dense+Reranker)']['typo_top1'],
           versions['v4 (BM25+ngram+Dense+Reranker)']['keyword_mrr'], v4_lat]

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

v2_vals += v2_vals[:1]
v3_vals += v3_vals[:1]
v4_vals += v4_vals[:1]
categories += categories[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, v2_vals, 'o-', linewidth=2, label='v2 (BM25+TF-IDF)', color='#3b82f6')
ax.fill(angles, v2_vals, alpha=0.1, color='#3b82f6')
ax.plot(angles, v3_vals, 's-', linewidth=2, label='v3 (BM25+Dense)', color='#10b981')
ax.fill(angles, v3_vals, alpha=0.1, color='#10b981')
ax.plot(angles, v4_vals, '^-', linewidth=2, label='v4 (BM25+ngram+Dense+Reranker)', color='#f59e0b')
ax.fill(angles, v4_vals, alpha=0.1, color='#f59e0b')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories[:-1], fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title('Retrieval Capability Radar: v2 vs v3 vs v4', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig(output_dir / 'version_comparison_radar.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 4: Keyword-only Improvement Detail
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Keyword-only Format: v2 → v3 → v4 Improvement', fontsize=14, fontweight='bold')

# MRR progression
ax = axes[0]
versions_list = ['v2', 'v3', 'v4']
mrr_vals = [0.180, 0.180, 0.333]
bars = ax.bar(versions_list, mrr_vals, color=colors, edgecolor='white', width=0.5)
ax.set_title('Keyword-only MRR Progression', fontweight='bold')
ax.set_ylabel('MRR')
ax.set_ylim(0, 0.5)
for bar, val in zip(bars, mrr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# Add improvement annotation
ax.annotate('+85%', xy=(2, 0.333), xytext=(1.5, 0.4),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, fontweight='bold', color='green')

# Hit@5 progression
ax = axes[1]
hit_vals = [0.40, 0.45, 0.33]  # v2, v3, v4
bars = ax.bar(versions_list, hit_vals, color=colors, edgecolor='white', width=0.5)
ax.set_title('Keyword-only Hit@5', fontweight='bold')
ax.set_ylabel('Hit Rate')
ax.set_ylim(0, 0.6)
for bar, val in zip(bars, hit_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.0%}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'keyword_only_improvement.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Plot 5: Latency Breakdown
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

stages_v3 = ['BM25 Search', 'Dense Encode', 'Dense Matmul', 'TF-IDF Cosine']
times_v3 = [10, 70, 5, 1800]  # ms

stages_v4 = ['BM25+ngram Search', 'Dense Encode', 'Dense Matmul', 'Cross-encoder Rerank']
times_v4 = [10, 70, 5, 1500]  # ms

x = np.arange(len(stages_v3))
width = 0.35

bars1 = ax.bar(x - width/2, times_v3, width, label='v3', color='#3b82f6', edgecolor='white')
bars2 = ax.bar(x + width/2, times_v4, width, label='v4', color='#f59e0b', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(stages_v3, rotation=15, ha='right')
ax.set_ylabel('Time (ms)')
ax.set_title('Per-Query Latency Breakdown: v3 vs v4', fontweight='bold', fontsize=14)
ax.set_yscale('log')
ax.legend(fontsize=10)

for bar, val in zip(bars1, times_v3):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{val}ms', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, times_v4):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, f'{val}ms', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add total
ax.text(0.5, 0.95, f'v3 Total: {sum(times_v3)}ms  |  v4 Total: {sum(times_v4)}ms (+{sum(times_v4)-sum(times_v3)}ms)',
        transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'latency_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()

print("All plots generated in data/plots/")
print("Files created:")
for f in output_dir.glob('*.png'):
    print(f"  - {f.name}")