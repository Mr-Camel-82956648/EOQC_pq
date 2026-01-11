#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似度分析可视化脚本
合并了 similarity_analysis.py 和 generate_median_analysis.py 的绘图功能
支持从 JSON 摘要或 CSV 文件生成可视化图表
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Dict, List, Optional

# ========== 中文字体配置 ==========
def setup_chinese_font():
    """设置中文字体，解决图表中文显示问题"""
    font_path = '/home/dataset-assist-0/rzh/projects/EOQC_pq/venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'
    
    if os.path.exists(font_path):
        try:
            chinese_font = fm.FontProperties(fname=font_path)
            try:
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            except:
                pass
            plt.rcParams['axes.unicode_minus'] = False
            print("✓ 已加载中文字体: SimHei")
            return chinese_font
        except Exception as e:
            print(f"⚠ 字体加载失败: {str(e)}")
            return None
    else:
        print(f"⚠ 字体文件不存在: {font_path}")
        print("  将使用默认字体，中文可能显示为方块")
        return None

chinese_font_prop = setup_chinese_font()
chinese_font_available = chinese_font_prop is not None

# ========== 配置 ==========
DIMENSIONS_TO_TEST = [32, 64, 128, 256, 512]

# ========== 数据加载函数 ==========
def load_summary_data(filepath: str) -> Dict:
    """从JSON文件加载统计摘要数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
    """从CSV文件加载详细结果数据"""
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"[警告] 加载CSV文件失败: {e}")
        return None

# ========== 从JSON摘要提取数据 ==========
def extract_median_data(summary: Dict) -> Dict[str, Dict[int, float]]:
    """从摘要中提取中位数数据，返回 {关系类型: {维度: 中位数}}"""
    median_data = {}
    
    for dim_str, dim_stats in summary['按维度的统计'].items():
        dim = int(dim_str)
        for rel_type, stats in dim_stats.items():
            if rel_type not in median_data:
                median_data[rel_type] = {}
            median_data[rel_type][dim] = stats['中位数']
    
    return median_data

def extract_mean_data(summary: Dict) -> Dict[str, Dict[int, float]]:
    """从摘要中提取均值数据，返回 {关系类型: {维度: 平均值}}"""
    mean_data = {}
    
    for dim_str, dim_stats in summary['按维度的统计'].items():
        dim = int(dim_str)
        for rel_type, stats in dim_stats.items():
            if rel_type not in mean_data:
                mean_data[rel_type] = {}
            mean_data[rel_type][dim] = stats['平均值']
    
    return mean_data

# ========== 从JSON摘要生成图表（原有功能）==========
def generate_median_analysis_charts(summary: Dict, output_dir: str, model_name: str):
    """生成中位数分析图表（4张图合并到一张）"""
    print("\n" + "="*70)
    print("开始生成中位数分析图表 (共4张)")
    print("="*70)
    
    median_data = extract_median_data(summary)
    mean_data = extract_mean_data(summary)
    relation_types = sorted(list(median_data.keys()))
    
    if chinese_font_available:
        font_name = chinese_font_prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 图1: 各维度下关系类型排序趋势（使用中位数）
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
                      label=f'{dim}维', linewidth=2, markersize=6)
    
    axes[0, 0].set_title('图1: 各维度下关系类型排序趋势（中位数）', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 0].set_xlabel('关系类型 (按512维中位数排序)', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 0].set_ylabel('中位数相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 0].set_xticks(range(len(type_order)))
    axes[0, 0].set_xticklabels(type_order, rotation=45, ha='right')
    if chinese_font_available:
        for label in axes[0, 0].get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in axes[0, 0].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    legend1 = axes[0, 0].legend()
    if chinese_font_available and legend1:
        for text in legend1.get_texts():
            text.set_fontproperties(chinese_font_prop)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图2: 各关系类型在不同维度下的中位数变化趋势
    for rel_type in relation_types:
        dim_medians = [median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        axes[0, 1].plot(DIMENSIONS_TO_TEST, dim_medians, marker='o', label=rel_type, 
                       linewidth=2, markersize=5)
    
    axes[0, 1].set_title('图2: 各关系类型在不同维度下的中位数变化', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 1].set_xlabel('嵌入维度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 1].set_ylabel('中位数相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 1].set_xticks(DIMENSIONS_TO_TEST)
    axes[0, 1].set_xticklabels(DIMENSIONS_TO_TEST)
    if chinese_font_available:
        for label in axes[0, 1].get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in axes[0, 1].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    legend2 = axes[0, 1].legend(fontsize=8, ncol=2, loc='best')
    if chinese_font_available and legend2:
        for text in legend2.get_texts():
            text.set_fontproperties(chinese_font_prop)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3: 各类型×维度的平均相似度热力图
    type_mean_avg = {}
    for rel_type in relation_types:
        mean_values = [mean_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        type_mean_avg[rel_type] = np.mean(mean_values)
    
    type_order_by_mean = sorted(type_mean_avg.keys(), key=lambda x: type_mean_avg[x], reverse=True)
    heatmap_data_mean = np.array([[mean_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST] 
                                   for rel_type in type_order_by_mean])
    
    im1 = axes[1, 0].imshow(heatmap_data_mean, aspect='auto', cmap='YlOrRd')
    axes[1, 0].set_title('图3: 各类型×维度的平均相似度热力图（按均值排序）', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 0].set_xlabel('嵌入维度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 0].set_ylabel('关系类型（按均值从高到低）', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 0].set_xticks(range(len(DIMENSIONS_TO_TEST)))
    axes[1, 0].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[1, 0].set_yticks(range(len(type_order_by_mean)))
    axes[1, 0].set_yticklabels(type_order_by_mean)
    if chinese_font_available:
        for label in axes[1, 0].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    cbar1 = plt.colorbar(im1, ax=axes[1, 0])
    if chinese_font_available:
        cbar1.set_label('平均相似度', fontproperties=chinese_font_prop)
    
    # 图4: 各类型×维度的中位数相似度热力图
    type_median_avg = {}
    for rel_type in relation_types:
        median_values = [median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        type_median_avg[rel_type] = np.mean(median_values)
    
    type_order_by_median = sorted(type_median_avg.keys(), key=lambda x: type_median_avg[x], reverse=True)
    heatmap_data_median = np.array([[median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST] 
                                     for rel_type in type_order_by_median])
    
    im2 = axes[1, 1].imshow(heatmap_data_median, aspect='auto', cmap='YlOrRd')
    axes[1, 1].set_title('图4: 各类型×维度的中位数相似度热力图（按中位数排序）', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 1].set_xlabel('嵌入维度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 1].set_ylabel('关系类型（按中位数从高到低）', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 1].set_xticks(range(len(DIMENSIONS_TO_TEST)))
    axes[1, 1].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[1, 1].set_yticks(range(len(type_order_by_median)))
    axes[1, 1].set_yticklabels(type_order_by_median)
    if chinese_font_available:
        for label in axes[1, 1].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    cbar2 = plt.colorbar(im2, ax=axes[1, 1])
    if chinese_font_available:
        cbar2.set_label('中位数相似度', fontproperties=chinese_font_prop)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'similarity_analysis_median.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 中位数分析图表已生成并保存为: {output_file}")

# ========== 从CSV DataFrame生成图表（从similarity_analysis.py提取）==========
def generate_visualizations(df: pd.DataFrame, output_dir: str, model_name: str):
    """生成可视化图表（从DataFrame生成6张图+维度箱型图）"""
    print("\n" + "="*70)
    print("开始生成可视化图表 (共6张+维度箱型图)")
    print("="*70)
    
    if not chinese_font_available:
        print("\n[注意] 中文字体加载失败，图表将使用英文标签。")
        en_labels = {
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
        df['关系类型_en'] = df['关系类型'].map(en_labels)
    
    if chinese_font_available:
        font_name = chinese_font_prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    
    # 1. 按关系类型的箱线图
    if not chinese_font_available:
        sns.boxplot(x='关系类型_en', y='余弦相似度', data=df, ax=axes[0, 0])
        axes[0, 0].set_xlabel('关系类型 (英文)')
    else:
        sns.boxplot(x='关系类型', y='余弦相似度', data=df, ax=axes[0, 0])
        axes[0, 0].set_xlabel('关系类型', fontproperties=chinese_font_prop)
    axes[0, 0].set_title('图1: 各关系类型的相似度分布', fontsize=14, fontweight='bold', 
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 0].set_ylabel('余弦相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 0].tick_params(axis='x', rotation=45)
    if chinese_font_available:
        for label in axes[0, 0].get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in axes[0, 0].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    
    # 2. 按维度的箱线图
    sns.boxplot(x='嵌入维度', y='余弦相似度', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('图2: 各嵌入维度的相似度分布', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 1].set_xlabel('嵌入维度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 1].set_ylabel('余弦相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
    if chinese_font_available:
        for label in axes[0, 1].get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in axes[0, 1].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    
    # 3. 维度与关系类型交互的热力图
    if not chinese_font_available:
        pivot_data = df.groupby(['关系类型_en', '嵌入维度'])['余弦相似度'].mean().unstack()
    else:
        pivot_data = df.groupby(['关系类型', '嵌入维度'])['余弦相似度'].mean().unstack()
    
    im = axes[0, 2].imshow(pivot_data, aspect='auto', cmap='YlOrRd')
    axes[0, 2].set_title('图3: 各类型×维度的平均相似度热力图', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 2].set_xlabel('嵌入维度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 2].set_ylabel('关系类型', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[0, 2].set_xticks(range(len(DIMENSIONS_TO_TEST)))
    axes[0, 2].set_xticklabels(DIMENSIONS_TO_TEST)
    axes[0, 2].set_yticks(range(len(pivot_data.index)))
    axes[0, 2].set_yticklabels(pivot_data.index)
    if chinese_font_available:
        for label in axes[0, 2].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    cbar = plt.colorbar(im, ax=axes[0, 2])
    if chinese_font_available:
        cbar.set_label('平均相似度', fontproperties=chinese_font_prop)
    
    # 4. 所有维度下关系类型趋势
    dim_512_df = df[df['嵌入维度'] == 512]
    if not chinese_font_available:
        type_means_512 = dim_512_df.groupby('关系类型_en')['余弦相似度'].mean().sort_values()
        type_order = type_means_512.index.tolist()
    else:
        type_means_512 = dim_512_df.groupby('关系类型')['余弦相似度'].mean().sort_values()
        type_order = type_means_512.index.tolist()
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    
    for i, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_df = df[df['嵌入维度'] == dim]
        if not chinese_font_available:
            type_means = dim_df.groupby('关系类型_en')['余弦相似度'].mean()
        else:
            type_means = dim_df.groupby('关系类型')['余弦相似度'].mean()
        
        type_means_ordered = [type_means.get(t, 0) for t in type_order]
        axes[1, 0].plot(type_means_ordered, marker=markers[i], color=colors[i], 
                      label=f'{dim}维', linewidth=2, markersize=6)
    
    axes[1, 0].set_title('图4: 各维度下关系类型排序趋势 (所有维度)', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 0].set_xlabel('关系类型 (按512维相似度排序)', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 0].set_ylabel('平均相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 0].set_xticks(range(len(type_order)))
    axes[1, 0].set_xticklabels(type_order, rotation=45, ha='right')
    if chinese_font_available:
        for label in axes[1, 0].get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in axes[1, 0].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    legend1 = axes[1, 0].legend()
    if chinese_font_available and legend1:
        for text in legend1.get_texts():
            text.set_fontproperties(chinese_font_prop)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 各关系类型在不同维度下的变化趋势
    if not chinese_font_available:
        relation_types = sorted(df['关系类型_en'].unique())
    else:
        relation_types = sorted(df['关系类型'].unique())
    
    for rel_type in relation_types:
        type_df = df[df['关系类型' if chinese_font_available else '关系类型_en'] == rel_type]
        dim_means = []
        for dim in DIMENSIONS_TO_TEST:
            dim_data = type_df[type_df['嵌入维度'] == dim]['余弦相似度']
            if len(dim_data) > 0:
                dim_means.append(dim_data.mean())
            else:
                dim_means.append(0)
        
        axes[1, 1].plot(DIMENSIONS_TO_TEST, dim_means, marker='o', label=rel_type, linewidth=2, markersize=5)
    
    axes[1, 1].set_title('图5: 各关系类型在不同维度下的相似度变化', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 1].set_xlabel('嵌入维度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 1].set_ylabel('平均相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 1].set_xticks(DIMENSIONS_TO_TEST)
    axes[1, 1].set_xticklabels(DIMENSIONS_TO_TEST)
    if chinese_font_available:
        for label in axes[1, 1].get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in axes[1, 1].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    legend2 = axes[1, 1].legend(fontsize=8, ncol=2, loc='best')
    if chinese_font_available and legend2:
        for text in legend2.get_texts():
            text.set_fontproperties(chinese_font_prop)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 所有相似度值的分布直方图
    axes[1, 2].hist(df['余弦相似度'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 2].axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='高难度 (0.8)')
    axes[1, 2].axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='中难度 (0.5)')
    axes[1, 2].axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='低难度 (0.2)')
    axes[1, 2].set_title('图6: 相似度整体分布与难度阈值', fontsize=14, fontweight='bold',
                        fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 2].set_xlabel('余弦相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
    axes[1, 2].set_ylabel('样本频数', fontproperties=chinese_font_prop if chinese_font_available else None)
    if chinese_font_available:
        for label in axes[1, 2].get_xticklabels():
            label.set_fontproperties(chinese_font_prop)
        for label in axes[1, 2].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    legend3 = axes[1, 2].legend()
    if chinese_font_available and legend3:
        for text in legend3.get_texts():
            text.set_fontproperties(chinese_font_prop)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'similarity_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n可视化图表已生成并保存为: {output_file}")
    
    # 生成每个维度的单独箱型图
    print("\n开始生成各维度的单独箱型图...")
    dimension_output_file = os.path.join(output_dir, 'similarity_analysis_by_dimension.png')
    
    fig_dim, axes_dim = plt.subplots(2, 3, figsize=(20, 13))
    axes_dim = axes_dim.flatten()
    
    for idx, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_df = df[df['嵌入维度'] == dim]
        if dim_df.empty:
            continue
        
        ax = axes_dim[idx]
        
        if not chinese_font_available:
            sns.boxplot(x='关系类型_en', y='余弦相似度', data=dim_df, ax=ax)
            ax.set_xlabel('关系类型 (英文)')
        else:
            sns.boxplot(x='关系类型', y='余弦相似度', data=dim_df, ax=ax)
            ax.set_xlabel('关系类型', fontproperties=chinese_font_prop)
        
        ax.set_title(f'{dim}维: 各关系类型的相似度分布', fontsize=14, fontweight='bold',
                    fontproperties=chinese_font_prop if chinese_font_available else None)
        ax.set_ylabel('余弦相似度', fontproperties=chinese_font_prop if chinese_font_available else None)
        ax.tick_params(axis='x', rotation=45)
        
        if chinese_font_available:
            for label in ax.get_xticklabels():
                label.set_fontproperties(chinese_font_prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(chinese_font_prop)
    
    if len(DIMENSIONS_TO_TEST) < 6:
        axes_dim[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(dimension_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"各维度的单独箱型图已生成并保存为: {dimension_output_file}")

# ========== 主函数 ==========
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='相似度分析可视化脚本')
    parser.add_argument('--model', type=str, required=True, help='模型名称，用于确定输出目录')
    
    args = parser.parse_args()
    
    model_name = args.model
    result_base_dir = 'result'
    output_dir = os.path.join(result_base_dir, model_name)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print(f"相似度分析可视化脚本")
    print(f"模型: {model_name}")
    print(f"输出目录: {output_dir}")
    print("="*70)
    
    # 默认文件路径
    default_json_file = 'similarity_summary_large_scale.json'
    default_csv_file = 'similarity_results_large_scale.csv'
    
    json_path = os.path.join(output_dir, default_json_file)
    csv_path = os.path.join(output_dir, default_csv_file)
    
    # 加载JSON摘要（如果存在）
    summary = None
    if os.path.exists(json_path):
        print(f"\n正在加载JSON摘要文件: {json_path}")
        try:
            summary = load_summary_data(json_path)
            print(f"✓ JSON摘要加载成功")
            print(f"  测试维度: {summary.get('测试的维度', 'N/A')}")
        except Exception as e:
            print(f"[警告] 加载JSON摘要失败: {e}")
    else:
        print(f"\n[提示] JSON摘要文件不存在: {json_path}")
        print(f"  将跳过中位数分析图表的生成")
    
    # 加载CSV数据（如果存在）
    df = None
    if os.path.exists(csv_path):
        print(f"\n正在加载CSV结果文件: {csv_path}")
        df = load_csv_data(csv_path)
        if df is not None:
            print(f"✓ CSV数据加载成功")
            print(f"  数据行数: {len(df)}")
    else:
        print(f"\n[提示] CSV结果文件不存在: {csv_path}")
        print(f"  将跳过详细可视化图表的生成")
    
    # 生成图表（如果数据存在）
    chart_generated = False
    
    if summary:
        try:
            generate_median_analysis_charts(summary, output_dir, model_name)
            chart_generated = True
        except Exception as e:
            print(f"[警告] 生成中位数分析图表时出错: {e}")
    
    if df is not None:
        try:
            generate_visualizations(df, output_dir, model_name)
            chart_generated = True
        except Exception as e:
            print(f"[警告] 生成可视化图表时出错: {e}")
    
    print("\n" + "="*70)
    if chart_generated:
        print("完成！")
    else:
        print("未生成任何图表（缺少数据文件）")
    print("="*70)

if __name__ == "__main__":
    main()

