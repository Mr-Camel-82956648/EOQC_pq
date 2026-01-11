#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成所有模型的英文版本图表（整合版）
自动扫描 result 目录下的所有模型目录，为每个模型生成英文版本的图表
整合了 similarity_visualize_en.py 和 single_dim_models_visualize_en.py 的所有功能
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

# 尝试导入 model_list，如果失败则使用空列表
try:
    from single_dim_models_analysis import model_list
except ImportError:
    model_list = []

# ========== Configuration ==========
DIMENSIONS_TO_TEST = [32, 64, 128, 256, 512]

# ========== Relation Type Mapping (Chinese to English) ==========
RELATION_TYPE_MAPPING = {
    '同义不同表达': 'Synonym Different Expression',
    '近义词': 'Synonym',
    '反义词': 'Antonym',
    '双重否定': 'Double Negation',
    '部分正确': 'Partially Correct',
    '无关项': 'Irrelevant',
    '句式变化': 'Sentence Structure',
    '概念混淆': 'Concept Confusion',
    '过度概括': 'Overgeneralization'
}

# Short labels for compact display
RELATION_TYPE_SHORT = {
    '同义不同表达': 'Synonym\nDiff Expr',
    '近义词': 'Synonym',
    '反义词': 'Antonym',
    '双重否定': 'Double\nNegation',
    '部分正确': 'Partially\nCorrect',
    '无关项': 'Irrelevant',
    '句式变化': 'Sentence\nStructure',
    '概念混淆': 'Concept\nConfusion',
    '过度概括': 'Overgeneralization'
}

def translate_relation_type(chinese_type: str, use_short: bool = False) -> str:
    """Translate Chinese relation type to English"""
    mapping = RELATION_TYPE_SHORT if use_short else RELATION_TYPE_MAPPING
    return mapping.get(chinese_type, chinese_type)

# ========== Data Loading Functions ==========
def load_summary_data(filepath: str) -> Dict:
    """Load statistical summary data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load detailed result data from CSV file"""
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"[Warning] Failed to load CSV file: {e}")
        return None

def translate_dataframe_relation_types(df: pd.DataFrame) -> pd.DataFrame:
    """Translate relation types in DataFrame from Chinese to English"""
    df = df.copy()
    if '关系类型' in df.columns:
        df['关系类型_en'] = df['关系类型'].map(lambda x: translate_relation_type(x, use_short=True))
    return df

def get_models_sorted(df: pd.DataFrame) -> List[str]:
    """Get sorted model list from DataFrame by dimension"""
    # Get actual models in DataFrame
    actual_models = df['模型名称'].unique().tolist()
    
    if '模型维度' in df.columns:
        # Sort by dimension, then by model name if dimension is the same
        model_dims = df.groupby('模型名称')['模型维度'].first().to_dict()
        # Only use models that exist in DataFrame
        if model_list:
            models_sorted = sorted(actual_models, key=lambda m: (model_dims.get(m, 999999), m))
        else:
            models_sorted = sorted(actual_models, key=lambda m: (model_dims.get(m, 999999), m))
    else:
        # If no dimension info, use original order
        models_sorted = sorted(actual_models)
    return models_sorted

# ========== Extract Data from JSON Summary (Multi-Dimension Models) ==========
def extract_median_data_by_dimension(summary: Dict) -> Dict[str, Dict[int, float]]:
    """Extract median data from summary, returns {relation_type: {dimension: median}}"""
    median_data = {}
    
    # Handle both Chinese and English keys
    stats_key = '按维度的统计' if '按维度的统计' in summary else 'by_dimension'
    dim_stats = summary.get(stats_key, {})
    
    for dim_str, dim_data in dim_stats.items():
        dim = int(dim_str)
        for rel_type, stats in dim_data.items():
            # Ensure English relation type
            if rel_type in RELATION_TYPE_MAPPING.values() or rel_type in RELATION_TYPE_SHORT.values():
                en_type = rel_type
            else:
                en_type = translate_relation_type(rel_type)
            
            if en_type not in median_data:
                median_data[en_type] = {}
            median_data[en_type][dim] = stats.get('中位数', stats.get('median', 0))
    
    return median_data

def extract_mean_data_by_dimension(summary: Dict) -> Dict[str, Dict[int, float]]:
    """Extract mean data from summary, returns {relation_type: {dimension: mean}}"""
    mean_data = {}
    
    # Handle both Chinese and English keys
    stats_key = '按维度的统计' if '按维度的统计' in summary else 'by_dimension'
    dim_stats = summary.get(stats_key, {})
    
    for dim_str, dim_data in dim_stats.items():
        dim = int(dim_str)
        for rel_type, stats in dim_data.items():
            # Ensure English relation type
            if rel_type in RELATION_TYPE_MAPPING.values() or rel_type in RELATION_TYPE_SHORT.values():
                en_type = rel_type
            else:
                en_type = translate_relation_type(rel_type)
            
            if en_type not in mean_data:
                mean_data[en_type] = {}
            mean_data[en_type][dim] = stats.get('平均值', stats.get('mean', 0))
    
    return mean_data

# ========== Extract Data from JSON Summary (Single-Dimension Models) ==========
def extract_median_data_by_model(summary: Dict) -> Dict[str, Dict[str, float]]:
    """Extract median data from summary, returns {relation_type: {model: median}}"""
    median_data = {}
    models_sorted = summary.get('测试的模型', summary.get('tested_models', model_list))
    
    # Handle both Chinese and English keys
    stats_key = '按模型的统计' if '按模型的统计' in summary else 'by_model'
    model_stats = summary.get(stats_key, {})
    
    for model, model_data in model_stats.items():
        for rel_type, stats in model_data.items():
            # Ensure English relation type
            if rel_type in RELATION_TYPE_MAPPING.values() or rel_type in RELATION_TYPE_SHORT.values():
                en_type = rel_type
            else:
                en_type = translate_relation_type(rel_type)
            
            if en_type not in median_data:
                median_data[en_type] = {}
            median_data[en_type][model] = stats.get('中位数', stats.get('median', 0))
    
    return median_data

def extract_mean_data_by_model(summary: Dict) -> Dict[str, Dict[str, float]]:
    """Extract mean data from summary, returns {relation_type: {model: mean}}"""
    mean_data = {}
    
    # Handle both Chinese and English keys
    stats_key = '按模型的统计' if '按模型的统计' in summary else 'by_model'
    model_stats = summary.get(stats_key, {})
    
    for model, model_data in model_stats.items():
        for rel_type, stats in model_data.items():
            # Ensure English relation type
            if rel_type in RELATION_TYPE_MAPPING.values() or rel_type in RELATION_TYPE_SHORT.values():
                en_type = rel_type
            else:
                en_type = translate_relation_type(rel_type)
            
            if en_type not in mean_data:
                mean_data[en_type] = {}
            mean_data[en_type][model] = stats.get('平均值', stats.get('mean', 0))
    
    return mean_data

# ========== Generate Charts for Multi-Dimension Models ==========
def generate_median_analysis_charts_by_dimension(summary: Dict, output_dir: str, model_name: str):
    """Generate median analysis charts for multi-dimension models (4 charts combined into one)"""
    print("\n" + "="*70)
    print("Generating median analysis charts (4 charts)")
    print("="*70)
    
    median_data = extract_median_data_by_dimension(summary)
    mean_data = extract_mean_data_by_dimension(summary)
    relation_types = sorted(list(median_data.keys()))
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Chart 1: Relation type ranking trends across dimensions (using median)
    dim_512_medians = {}
    for rel_type in relation_types:
        if 512 in median_data[rel_type]:
            dim_512_medians[rel_type] = median_data[rel_type][512]
    
    type_order = sorted(dim_512_medians.keys(), key=lambda x: dim_512_medians[x])
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    
    for i, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_medians_ordered = [median_data[rel_type].get(dim, 0) for rel_type in type_order]
        axes[0, 0].plot(dim_medians_ordered, marker=markers[i], color=colors[i], 
                      label=f'{dim}D', linewidth=2, markersize=6)
    
    axes[0, 0].set_title('Chart 1: Relation Type Ranking Trends by Dimension (Median)', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Relation Type (Sorted by 512D Median)', fontsize=11)
    axes[0, 0].set_ylabel('Median Similarity', fontsize=11)
    axes[0, 0].set_xticks(range(len(type_order)))
    axes[0, 0].set_xticklabels(type_order, rotation=45, ha='right', fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Chart 2: Median changes of each relation type across dimensions
    for rel_type in relation_types:
        dim_medians = [median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        axes[0, 1].plot(DIMENSIONS_TO_TEST, dim_medians, marker='o', label=rel_type, 
                       linewidth=2, markersize=5)
    
    axes[0, 1].set_title('Chart 2: Median Changes of Relation Types Across Dimensions', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Embedding Dimension', fontsize=11)
    axes[0, 1].set_ylabel('Median Similarity', fontsize=11)
    axes[0, 1].set_xticks(DIMENSIONS_TO_TEST)
    axes[0, 1].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[0, 1].legend(fontsize=8, ncol=2, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Chart 3: Mean similarity heatmap (type × dimension)
    type_mean_avg = {}
    for rel_type in relation_types:
        mean_values = [mean_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        type_mean_avg[rel_type] = np.mean(mean_values)
    
    type_order_by_mean = sorted(type_mean_avg.keys(), key=lambda x: type_mean_avg[x], reverse=True)
    heatmap_data_mean = np.array([[mean_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST] 
                                   for rel_type in type_order_by_mean])
    
    im1 = axes[1, 0].imshow(heatmap_data_mean, aspect='auto', cmap='YlOrRd')
    axes[1, 0].set_title('Chart 3: Mean Similarity Heatmap (Type × Dimension, Sorted by Mean)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Embedding Dimension', fontsize=11)
    axes[1, 0].set_ylabel('Relation Type (Sorted by Mean, High to Low)', fontsize=11)
    axes[1, 0].set_xticks(range(len(DIMENSIONS_TO_TEST)))
    axes[1, 0].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[1, 0].set_yticks(range(len(type_order_by_mean)))
    axes[1, 0].set_yticklabels(type_order_by_mean, fontsize=9)
    cbar1 = plt.colorbar(im1, ax=axes[1, 0])
    cbar1.set_label('Mean Similarity', fontsize=10)
    
    # Chart 4: Median similarity heatmap (type × dimension)
    type_median_avg = {}
    for rel_type in relation_types:
        median_values = [median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        type_median_avg[rel_type] = np.mean(median_values)
    
    type_order_by_median = sorted(type_median_avg.keys(), key=lambda x: type_median_avg[x], reverse=True)
    heatmap_data_median = np.array([[median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST] 
                                     for rel_type in type_order_by_median])
    
    im2 = axes[1, 1].imshow(heatmap_data_median, aspect='auto', cmap='YlOrRd')
    axes[1, 1].set_title('Chart 4: Median Similarity Heatmap (Type × Dimension, Sorted by Median)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Embedding Dimension', fontsize=11)
    axes[1, 1].set_ylabel('Relation Type (Sorted by Median, High to Low)', fontsize=11)
    axes[1, 1].set_xticks(range(len(DIMENSIONS_TO_TEST)))
    axes[1, 1].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[1, 1].set_yticks(range(len(type_order_by_median)))
    axes[1, 1].set_yticklabels(type_order_by_median, fontsize=9)
    cbar2 = plt.colorbar(im2, ax=axes[1, 1])
    cbar2.set_label('Median Similarity', fontsize=10)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'similarity_analysis_median.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Median analysis charts generated and saved as: {output_file}")

def generate_visualizations_by_dimension(df: pd.DataFrame, output_dir: str, model_name: str):
    """Generate visualization charts for multi-dimension models (6 charts + dimension boxplots)"""
    print("\n" + "="*70)
    print("Generating visualization charts (6 charts + dimension boxplots)")
    print("="*70)
    
    # Translate relation types
    df = translate_dataframe_relation_types(df)
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    
    # 1. Boxplot by relation type
    sns.boxplot(x='关系类型_en', y='余弦相似度', data=df, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Relation Type', fontsize=11)
    axes[0, 0].set_title('Chart 1: Similarity Distribution by Relation Type', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Cosine Similarity', fontsize=11)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Boxplot by dimension
    sns.boxplot(x='嵌入维度', y='余弦相似度', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Chart 2: Similarity Distribution by Embedding Dimension', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Embedding Dimension', fontsize=11)
    axes[0, 1].set_ylabel('Cosine Similarity', fontsize=11)
    
    # 3. Heatmap of dimension × relation type interaction
    pivot_data = df.groupby(['关系类型_en', '嵌入维度'])['余弦相似度'].mean().unstack()
    
    im = axes[0, 2].imshow(pivot_data, aspect='auto', cmap='YlOrRd')
    axes[0, 2].set_title('Chart 3: Mean Similarity Heatmap (Type × Dimension)', 
                        fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Embedding Dimension', fontsize=11)
    axes[0, 2].set_ylabel('Relation Type', fontsize=11)
    axes[0, 2].set_xticks(range(len(DIMENSIONS_TO_TEST)))
    axes[0, 2].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[0, 2].set_yticks(range(len(pivot_data.index)))
    axes[0, 2].set_yticklabels(pivot_data.index, fontsize=9)
    cbar = plt.colorbar(im, ax=axes[0, 2])
    cbar.set_label('Mean Similarity', fontsize=10)
    
    # 4. Relation type trends across all dimensions
    dim_512_df = df[df['嵌入维度'] == 512]
    type_means_512 = dim_512_df.groupby('关系类型_en')['余弦相似度'].mean().sort_values()
    type_order = type_means_512.index.tolist()
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    
    for i, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_df = df[df['嵌入维度'] == dim]
        type_means = dim_df.groupby('关系类型_en')['余弦相似度'].mean()
        type_means_ordered = [type_means.get(t, 0) for t in type_order]
        axes[1, 0].plot(type_means_ordered, marker=markers[i], color=colors[i], 
                      label=f'{dim}D', linewidth=2, markersize=6)
    
    axes[1, 0].set_title('Chart 4: Relation Type Ranking Trends Across Dimensions', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Relation Type (Sorted by 512D Similarity)', fontsize=11)
    axes[1, 0].set_ylabel('Mean Similarity', fontsize=11)
    axes[1, 0].set_xticks(range(len(type_order)))
    axes[1, 0].set_xticklabels(type_order, rotation=45, ha='right', fontsize=9)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Changes of each relation type across dimensions
    relation_types = sorted(df['关系类型_en'].unique())
    
    for rel_type in relation_types:
        type_df = df[df['关系类型_en'] == rel_type]
        dim_means = []
        for dim in DIMENSIONS_TO_TEST:
            dim_data = type_df[type_df['嵌入维度'] == dim]['余弦相似度']
            if len(dim_data) > 0:
                dim_means.append(dim_data.mean())
            else:
                dim_means.append(0)
        
        axes[1, 1].plot(DIMENSIONS_TO_TEST, dim_means, marker='o', label=rel_type, 
                       linewidth=2, markersize=5)
    
    axes[1, 1].set_title('Chart 5: Similarity Changes of Relation Types Across Dimensions', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Embedding Dimension', fontsize=11)
    axes[1, 1].set_ylabel('Mean Similarity', fontsize=11)
    axes[1, 1].set_xticks(DIMENSIONS_TO_TEST)
    axes[1, 1].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[1, 1].legend(fontsize=8, ncol=2, loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Distribution histogram of all similarity values
    axes[1, 2].hist(df['余弦相似度'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 2].axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='High Difficulty (0.8)')
    axes[1, 2].axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Medium Difficulty (0.5)')
    axes[1, 2].axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='Low Difficulty (0.2)')
    axes[1, 2].set_title('Chart 6: Overall Similarity Distribution and Difficulty Thresholds', 
                        fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Cosine Similarity', fontsize=11)
    axes[1, 2].set_ylabel('Sample Frequency', fontsize=11)
    axes[1, 2].legend()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'similarity_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization charts generated and saved as: {output_file}")
    
    # Generate separate boxplots for each dimension
    print("\nGenerating separate boxplots for each dimension...")
    dimension_output_file = os.path.join(output_dir, 'similarity_analysis_by_dimension.png')
    
    fig_dim, axes_dim = plt.subplots(2, 3, figsize=(20, 13))
    axes_dim = axes_dim.flatten()
    
    for idx, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_df = df[df['嵌入维度'] == dim]
        if dim_df.empty:
            continue
        
        ax = axes_dim[idx]
        sns.boxplot(x='关系类型_en', y='余弦相似度', data=dim_df, ax=ax)
        ax.set_xlabel('Relation Type', fontsize=10)
        ax.set_title(f'{dim}D: Similarity Distribution by Relation Type', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Cosine Similarity', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    
    if len(DIMENSIONS_TO_TEST) < 6:
        axes_dim[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(dimension_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Separate boxplots by dimension generated and saved as: {dimension_output_file}")

# ========== Generate Charts for Single-Dimension Models ==========
def generate_median_analysis_charts_by_model(summary: Dict, output_dir: str):
    """Generate median analysis charts for single-dimension models (4 charts combined into one)"""
    print("\n" + "="*70)
    print("Generating median analysis charts (4 charts)")
    print("="*70)
    
    median_data = extract_median_data_by_model(summary)
    mean_data = extract_mean_data_by_model(summary)
    models_sorted = summary.get('测试的模型', summary.get('tested_models', model_list))
    relation_types = sorted(list(median_data.keys()))
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    
    # Chart 1: Relation type ranking trends by model (using median)
    last_model = models_sorted[-1] if models_sorted else (model_list[-1] if model_list else '')
    last_model_medians = {}
    for rel_type in relation_types:
        if last_model in median_data[rel_type]:
            last_model_medians[rel_type] = median_data[rel_type][last_model]
    
    type_order = sorted(last_model_medians.keys(), key=lambda x: last_model_medians[x])
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', '+', 'x']
    colors = plt.cm.tab20(np.linspace(0, 1, len(models_sorted)))
    
    model_dim_mapping = summary.get('模型维度映射', summary.get('model_dimension_mapping', {}))
    
    for i, model in enumerate(models_sorted):
        if i < len(markers):
            marker = markers[i % len(markers)]
        else:
            marker = 'o'
        dim = model_dim_mapping.get(model, 0)
        model_label = f"{model}\n({dim}D)" if dim > 0 else model
        dim_medians_ordered = [median_data[rel_type].get(model, 0) for rel_type in type_order]
        axes[0, 0].plot(dim_medians_ordered, marker=marker, color=colors[i], 
                      label=model_label, linewidth=2, markersize=6)
    
    axes[0, 0].set_title(f'Chart 1: Relation Type Ranking Trends by Model (Median, Sorted by {last_model})', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Relation Type', fontsize=11)
    axes[0, 0].set_ylabel('Median Similarity', fontsize=11)
    axes[0, 0].set_xticks(range(len(type_order)))
    axes[0, 0].set_xticklabels(type_order, rotation=45, ha='right', fontsize=8)
    axes[0, 0].legend(fontsize=7, ncol=2, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Chart 2: Median changes of each relation type across models
    model_indices = list(range(len(models_sorted)))
    for rel_type in relation_types:
        model_medians = [median_data[rel_type].get(model, 0) for model in models_sorted]
        axes[0, 1].plot(model_indices, model_medians, marker='o', label=rel_type, 
                       linewidth=2, markersize=5)
    
    axes[0, 1].set_title('Chart 2: Median Changes of Relation Types Across Models', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Model (Sorted by Dimension, Small to Large)', fontsize=11)
    axes[0, 1].set_ylabel('Median Similarity', fontsize=11)
    axes[0, 1].set_xticks(model_indices)
    axes[0, 1].set_xticklabels([f"{m}\n({model_dim_mapping.get(m, 0)}D)" 
                                if model_dim_mapping.get(m, 0) > 0 else m 
                                for m in models_sorted], 
                               rotation=45, ha='right', fontsize=7)
    axes[0, 1].legend(fontsize=8, ncol=2, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Chart 3: Mean similarity heatmap (type × model)
    type_mean_avg = {}
    for rel_type in relation_types:
        mean_values = [mean_data[rel_type].get(model, 0) for model in models_sorted]
        type_mean_avg[rel_type] = np.mean(mean_values)
    
    type_order_by_mean = sorted(type_mean_avg.keys(), key=lambda x: type_mean_avg[x], reverse=True)
    heatmap_data_mean = np.array([[mean_data[rel_type].get(model, 0) for model in models_sorted] 
                                   for rel_type in type_order_by_mean])
    
    im1 = axes[1, 0].imshow(heatmap_data_mean, aspect='auto', cmap='YlOrRd')
    axes[1, 0].set_title('Chart 3: Mean Similarity Heatmap (Type × Model, Sorted by Mean)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Model (Sorted by Dimension, Small to Large)', fontsize=11)
    axes[1, 0].set_ylabel('Relation Type (Sorted by Mean, High to Low)', fontsize=11)
    axes[1, 0].set_xticks(range(len(models_sorted)))
    axes[1, 0].set_xticklabels([f"{m}\n({model_dim_mapping.get(m, 0)}D)" 
                                if model_dim_mapping.get(m, 0) > 0 else m 
                                for m in models_sorted], 
                               rotation=45, ha='right', fontsize=7)
    axes[1, 0].set_yticks(range(len(type_order_by_mean)))
    axes[1, 0].set_yticklabels(type_order_by_mean, fontsize=9)
    cbar1 = plt.colorbar(im1, ax=axes[1, 0])
    cbar1.set_label('Mean Similarity', fontsize=10)
    
    # Chart 4: Median similarity heatmap (type × model)
    type_median_avg = {}
    for rel_type in relation_types:
        median_values = [median_data[rel_type].get(model, 0) for model in models_sorted]
        type_median_avg[rel_type] = np.mean(median_values)
    
    type_order_by_median = sorted(type_median_avg.keys(), key=lambda x: type_median_avg[x], reverse=True)
    heatmap_data_median = np.array([[median_data[rel_type].get(model, 0) for model in models_sorted] 
                                     for rel_type in type_order_by_median])
    
    im2 = axes[1, 1].imshow(heatmap_data_median, aspect='auto', cmap='YlOrRd')
    axes[1, 1].set_title('Chart 4: Median Similarity Heatmap (Type × Model, Sorted by Median)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Model (Sorted by Dimension, Small to Large)', fontsize=11)
    axes[1, 1].set_ylabel('Relation Type (Sorted by Median, High to Low)', fontsize=11)
    axes[1, 1].set_xticks(range(len(models_sorted)))
    axes[1, 1].set_xticklabels([f"{m}\n({model_dim_mapping.get(m, 0)}D)" 
                                if model_dim_mapping.get(m, 0) > 0 else m 
                                for m in models_sorted], 
                               rotation=45, ha='right', fontsize=7)
    axes[1, 1].set_yticks(range(len(type_order_by_median)))
    axes[1, 1].set_yticklabels(type_order_by_median, fontsize=9)
    cbar2 = plt.colorbar(im2, ax=axes[1, 1])
    cbar2.set_label('Median Similarity', fontsize=10)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'similarity_analysis_median.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Median analysis charts generated and saved as: {output_file}")

def generate_visualizations_by_model(df: pd.DataFrame, output_dir: str):
    """Generate visualization charts for single-dimension models (6 charts + model boxplots)"""
    print("\n" + "="*70)
    print("Generating visualization charts (6 charts + model boxplots)")
    print("="*70)
    
    models_sorted = get_models_sorted(df)
    
    # Translate relation types
    df = translate_dataframe_relation_types(df)
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    
    # 1. Boxplot by relation type
    sns.boxplot(x='关系类型_en', y='余弦相似度', data=df, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Relation Type', fontsize=11)
    axes[0, 0].set_title('Chart 1: Similarity Distribution by Relation Type', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Cosine Similarity', fontsize=11)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Boxplot by model
    # Only use models that actually exist in DataFrame
    actual_models_in_df = df['模型名称'].unique().tolist()
    models_sorted_filtered = [m for m in models_sorted if m in actual_models_in_df]
    
    df['模型名称_ordered'] = pd.Categorical(df['模型名称'], categories=models_sorted_filtered, ordered=True)
    df_sorted = df.sort_values('模型名称_ordered')
    
    sns.boxplot(x='模型名称_ordered', y='余弦相似度', data=df_sorted, ax=axes[0, 1])
    axes[0, 1].set_title('Chart 2: Similarity Distribution by Model', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Model (Sorted by Dimension, Small to Large)', fontsize=11)
    axes[0, 1].set_ylabel('Cosine Similarity', fontsize=11)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Heatmap of model × relation type interaction
    pivot_data = df.groupby(['关系类型_en', '模型名称_ordered'])['余弦相似度'].mean().unstack()
    pivot_data = pivot_data.reindex(columns=models_sorted_filtered, fill_value=0)
    
    im = axes[0, 2].imshow(pivot_data, aspect='auto', cmap='YlOrRd')
    axes[0, 2].set_title('Chart 3: Mean Similarity Heatmap (Type × Model)', 
                        fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Model (Sorted by Dimension, Small to Large)', fontsize=11)
    axes[0, 2].set_ylabel('Relation Type', fontsize=11)
    axes[0, 2].set_xticks(range(len(models_sorted_filtered)))
    axes[0, 2].set_xticklabels(models_sorted_filtered, rotation=45, ha='right', fontsize=7)
    axes[0, 2].set_yticks(range(len(pivot_data.index)))
    axes[0, 2].set_yticklabels(pivot_data.index, fontsize=9)
    cbar = plt.colorbar(im, ax=axes[0, 2])
    cbar.set_label('Mean Similarity', fontsize=10)
    
    # 4. Relation type trends across all models
    last_model = models_sorted_filtered[-1] if models_sorted_filtered else ''
    last_model_df = df[df['模型名称'] == last_model]
    type_means_last = last_model_df.groupby('关系类型_en')['余弦相似度'].mean().sort_values()
    type_order = type_means_last.index.tolist()
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', '+', 'x']
    colors = plt.cm.tab20(np.linspace(0, 1, len(models_sorted_filtered)))
    
    for i, model in enumerate(models_sorted_filtered):
        model_df = df[df['模型名称'] == model]
        type_means = model_df.groupby('关系类型_en')['余弦相似度'].mean()
        type_means_ordered = [type_means.get(t, 0) for t in type_order]
        marker = markers[i % len(markers)] if i < len(markers) else 'o'
        dim = df[df['模型名称'] == model]['模型维度'].iloc[0] if '模型维度' in df.columns else 0
        model_label = f"{model}\n({int(dim)}D)" if dim > 0 else model
        axes[1, 0].plot(type_means_ordered, marker=marker, color=colors[i], 
                      label=model_label, linewidth=2, markersize=6)
    
    axes[1, 0].set_title(f'Chart 4: Relation Type Ranking Trends by Model (Sorted by {last_model})', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Relation Type', fontsize=11)
    axes[1, 0].set_ylabel('Mean Similarity', fontsize=11)
    axes[1, 0].set_xticks(range(len(type_order)))
    axes[1, 0].set_xticklabels(type_order, rotation=45, ha='right', fontsize=8)
    axes[1, 0].legend(fontsize=7, ncol=2, loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Changes of each relation type across models
    relation_types = sorted(df['关系类型_en'].unique())
    
    model_indices = list(range(len(models_sorted_filtered)))
    for rel_type in relation_types:
        type_df = df[df['关系类型_en'] == rel_type]
        model_means = []
        for model in models_sorted_filtered:
            model_data = type_df[type_df['模型名称'] == model]['余弦相似度']
            if len(model_data) > 0:
                model_means.append(model_data.mean())
            else:
                model_means.append(0)
        
        axes[1, 1].plot(model_indices, model_means, marker='o', label=rel_type, 
                       linewidth=2, markersize=5)
    
    axes[1, 1].set_title('Chart 5: Similarity Changes of Relation Types Across Models', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Model (Sorted by Dimension, Small to Large)', fontsize=11)
    axes[1, 1].set_ylabel('Mean Similarity', fontsize=11)
    axes[1, 1].set_xticks(model_indices)
    axes[1, 1].set_xticklabels([f"{m}\n({int(df[df['模型名称']==m]['模型维度'].iloc[0])}D)" 
                                if '模型维度' in df.columns and len(df[df['模型名称']==m]) > 0 
                                else m for m in models_sorted_filtered], 
                               rotation=45, ha='right', fontsize=7)
    axes[1, 1].legend(fontsize=8, ncol=2, loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Distribution histogram of all similarity values
    axes[1, 2].hist(df['余弦相似度'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 2].axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='High Difficulty (0.8)')
    axes[1, 2].axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Medium Difficulty (0.5)')
    axes[1, 2].axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='Low Difficulty (0.2)')
    axes[1, 2].set_title('Chart 6: Overall Similarity Distribution and Difficulty Thresholds', 
                        fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Cosine Similarity', fontsize=11)
    axes[1, 2].set_ylabel('Sample Frequency', fontsize=11)
    axes[1, 2].legend()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'similarity_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization charts generated and saved as: {output_file}")
    
    # Generate separate boxplots for each model
    print("\nGenerating separate boxplots for each model...")
    model_output_file = os.path.join(output_dir, 'similarity_analysis_by_model.png')
    
    num_models = len(models_sorted_filtered)
    n_cols = 3
    n_rows = (num_models + n_cols - 1) // n_cols
    
    fig_model, axes_model = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes_model = axes_model.reshape(1, -1)
    axes_model = axes_model.flatten()
    
    for idx, model in enumerate(models_sorted_filtered):
        model_df = df[df['模型名称'] == model]
        if model_df.empty:
            continue
        
        ax = axes_model[idx]
        sns.boxplot(x='关系类型_en', y='余弦相似度', data=model_df, ax=ax)
        ax.set_xlabel('Relation Type', fontsize=10)
        
        dim = model_df['模型维度'].iloc[0] if '模型维度' in model_df.columns else 0
        title = f'{model} ({int(dim)}D): Similarity Distribution by Relation Type' if dim > 0 else f'{model}: Similarity Distribution by Relation Type'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Cosine Similarity', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide extra subplots
    for idx in range(len(models_sorted_filtered), len(axes_model)):
        axes_model[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(model_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Separate boxplots by model generated and saved as: {model_output_file}")

# ========== Main Functions ==========
def find_model_directories(result_dir: str) -> list:
    """Find all model directories containing data files"""
    model_dirs = []
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"[Error] Result directory does not exist: {result_dir}")
        return []
    
    # Find all subdirectories
    for item in result_path.iterdir():
        if item.is_dir() and item.name != 'ens':  # Exclude ens directory itself
            # Check if contains data files
            json_file = item / 'similarity_summary_large_scale.json'
            csv_file = item / 'similarity_results_large_scale.csv'
            
            if json_file.exists() or csv_file.exists():
                model_dirs.append(item.name)
    
    return sorted(model_dirs)

def process_multi_dimension_model(model_name: str, result_base_dir: str):
    """Process a single multi-dimension model"""
    print(f"\n{'='*70}")
    print(f"Processing model: {model_name}")
    print(f"{'='*70}")
    
    output_dir = os.path.join(result_base_dir, 'ens', model_name)
    source_dir = os.path.join(result_base_dir, model_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default file paths
    default_json_file = 'similarity_summary_large_scale.json'
    default_csv_file = 'similarity_results_large_scale.csv'
    
    json_path = os.path.join(source_dir, default_json_file)
    csv_path = os.path.join(source_dir, default_csv_file)
    
    # Load JSON summary (if exists)
    summary = None
    if os.path.exists(json_path):
        print(f"\nLoading JSON summary file: {json_path}")
        try:
            summary = load_summary_data(json_path)
            print(f"✓ JSON summary loaded successfully")
            dims_key = '测试的维度' if '测试的维度' in summary else 'tested_dimensions'
            print(f"  Tested dimensions: {summary.get(dims_key, 'N/A')}")
        except Exception as e:
            print(f"[Warning] Failed to load JSON summary: {e}")
    else:
        print(f"\n[Info] JSON summary file does not exist: {json_path}")
        print(f"  Will skip median analysis chart generation")
    
    # Load CSV data (if exists)
    df = None
    if os.path.exists(csv_path):
        print(f"\nLoading CSV result file: {csv_path}")
        df = load_csv_data(csv_path)
        if df is not None:
            print(f"✓ CSV data loaded successfully")
            print(f"  Number of rows: {len(df)}")
    else:
        print(f"\n[Info] CSV result file does not exist: {csv_path}")
        print(f"  Will skip detailed visualization chart generation")
    
    # Generate charts (if data exists)
    chart_generated = False
    
    if summary:
        try:
            generate_median_analysis_charts_by_dimension(summary, output_dir, model_name)
            chart_generated = True
        except Exception as e:
            print(f"[Warning] Error generating median analysis charts: {e}")
            import traceback
            traceback.print_exc()
    
    if df is not None:
        try:
            generate_visualizations_by_dimension(df, output_dir, model_name)
            chart_generated = True
        except Exception as e:
            print(f"[Warning] Error generating visualization charts: {e}")
            import traceback
            traceback.print_exc()
    
    return chart_generated

def process_single_dimension_models(result_base_dir: str):
    """Process single-dimension models"""
    print(f"\n{'='*70}")
    print("Processing single-dimension models")
    print(f"{'='*70}")
    
    output_dir = os.path.join(result_base_dir, 'ens', 'single_dim_models')
    source_dir = os.path.join(result_base_dir, 'single_dim_models')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default file paths
    default_json_file = 'similarity_summary_single_dim_models.json'
    default_csv_file = 'similarity_results_single_dim_models.csv'
    
    json_path = os.path.join(source_dir, default_json_file)
    csv_path = os.path.join(source_dir, default_csv_file)
    
    # Load JSON summary (if exists)
    summary = None
    if os.path.exists(json_path):
        print(f"\nLoading JSON summary file: {json_path}")
        try:
            summary = load_summary_data(json_path)
            print(f"✓ JSON summary loaded successfully")
            models_key = '测试的模型' if '测试的模型' in summary else 'tested_models'
            print(f"  Tested models: {summary.get(models_key, 'N/A')}")
        except Exception as e:
            print(f"[Warning] Failed to load JSON summary: {e}")
    else:
        print(f"\n[Info] JSON summary file does not exist: {json_path}")
        print(f"  Will skip median analysis chart generation")
    
    # Load CSV data (if exists)
    df = None
    if os.path.exists(csv_path):
        print(f"\nLoading CSV result file: {csv_path}")
        df = load_csv_data(csv_path)
        if df is not None:
            print(f"✓ CSV data loaded successfully")
            print(f"  Number of rows: {len(df)}")
    else:
        print(f"\n[Info] CSV result file does not exist: {csv_path}")
        print(f"  Will skip detailed visualization chart generation")
    
    # Generate charts (if data exists)
    chart_generated = False
    
    if summary:
        try:
            generate_median_analysis_charts_by_model(summary, output_dir)
            chart_generated = True
        except Exception as e:
            print(f"[Warning] Error generating median analysis charts: {e}")
            import traceback
            traceback.print_exc()
    
    if df is not None:
        try:
            generate_visualizations_by_model(df, output_dir)
            chart_generated = True
        except Exception as e:
            print(f"[Warning] Error generating visualization charts: {e}")
            import traceback
            traceback.print_exc()
    
    return chart_generated

def main():
    result_base_dir = 'result'
    
    print("="*70)
    print("Batch English Chart Generation Script (Integrated Version)")
    print("="*70)
    
    # Find all model directories
    print(f"\nScanning {result_base_dir} directory...")
    model_dirs = find_model_directories(result_base_dir)
    
    if not model_dirs:
        print(f"\n[Info] No model directories with data files found")
    else:
        print(f"\nFound {len(model_dirs)} model directories:")
        for model in model_dirs:
            print(f"  - {model}")
    
    # Process each multi-dimension model
    success_count = 0
    failed_models = []
    
    for model_name in model_dirs:
        if process_multi_dimension_model(model_name, result_base_dir):
            success_count += 1
        else:
            failed_models.append(model_name)
    
    # Process single-dimension models
    single_dim_success = process_single_dimension_models(result_base_dir)
    
    # Summary
    print("\n" + "="*70)
    print("Generation Complete!")
    print("="*70)
    print(f"\nSuccessfully generated: {success_count}/{len(model_dirs)} multi-dimension model charts")
    if single_dim_success:
        print("✓ Single-dimension model charts generated")
    else:
        print("✗ Single-dimension model charts generation failed")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    print(f"\nAll English charts saved to: {result_base_dir}/ens/")

if __name__ == "__main__":
    main()
