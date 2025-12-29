import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List

# ========== 中文字体配置 ==========
def setup_chinese_font():
    """设置中文字体，解决图表中文显示问题"""
    font_path = '/home/dataset-assist-0/rzh/projects/EOQC_pq/venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'
    
    if os.path.exists(font_path):
        try:
            # 创建FontProperties对象，直接指定字体文件（最可靠的方法）
            chinese_font = fm.FontProperties(fname=font_path)
            
            # 同时尝试注册到字体管理器（用于rcParams）
            try:
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            except:
                pass  # 如果注册失败，仍然可以使用FontProperties对象
            
            plt.rcParams['axes.unicode_minus'] = False
            print("✓ 已加载中文字体: SimHei")
            return chinese_font  # 返回FontProperties对象
        except Exception as e:
            print(f"⚠ 字体加载失败: {str(e)}")
            return None
    else:
        print(f"⚠ 字体文件不存在: {font_path}")
        print("  将使用默认字体，中文可能显示为方块")
        return None

# 执行字体设置，获取FontProperties对象
chinese_font_prop = setup_chinese_font()
chinese_font_available = chinese_font_prop is not None

# ========== 数据文件配置 ==========
SUMMARY_JSON_FILE = "similarity_summary_large_scale.json"
OUTPUT_FILE = "similarity_analysis_median.png"
DIMENSIONS_TO_TEST = [32, 64, 128, 256, 512]

def load_summary_data(filepath: str) -> Dict:
    """从JSON文件加载统计摘要数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

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

def generate_analysis_charts(summary: Dict, output_file: str = OUTPUT_FILE):
    """生成4张分析图表并合并到一张PNG"""
    print("\n开始生成中位数分析图表 (共4张)")
    print("="*70)
    
    # 提取数据
    median_data = extract_median_data(summary)
    mean_data = extract_mean_data(summary)
    
    # 获取所有关系类型
    relation_types = sorted(list(median_data.keys()))
    
    # 设置全局字体（如果可用）
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
    
    # ========== 图1: 各维度下关系类型排序趋势（使用中位数）==========
    # 使用512维的中位数排序作为基准
    dim_512_medians = {}
    for rel_type in relation_types:
        if 512 in median_data[rel_type]:
            dim_512_medians[rel_type] = median_data[rel_type][512]
    
    # 按512维的中位数排序
    type_order = sorted(dim_512_medians.keys(), key=lambda x: dim_512_medians[x])
    
    # 为不同维度设置不同的标记和颜色
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    
    for i, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_medians_ordered = []
        for rel_type in type_order:
            median_val = median_data[rel_type].get(dim, 0)
            dim_medians_ordered.append(median_val)
        
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
    
    # ========== 图2: 各关系类型在不同维度下的中位数变化趋势 ==========
    for rel_type in relation_types:
        dim_medians = []
        for dim in DIMENSIONS_TO_TEST:
            median_val = median_data[rel_type].get(dim, 0)
            dim_medians.append(median_val)
        
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
    
    # ========== 图3: 各类型×维度的平均相似度热力图（按均值从高到低排序）==========
    # 计算每个关系类型在所有维度下的平均值的平均值，用于排序
    type_mean_avg = {}
    for rel_type in relation_types:
        mean_values = [mean_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        type_mean_avg[rel_type] = np.mean(mean_values)
    
    # 按均值从高到低排序
    type_order_by_mean = sorted(type_mean_avg.keys(), key=lambda x: type_mean_avg[x], reverse=True)
    
    # 构建热力图数据矩阵
    heatmap_data_mean = []
    for rel_type in type_order_by_mean:
        row = [mean_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        heatmap_data_mean.append(row)
    heatmap_data_mean = np.array(heatmap_data_mean)
    
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
    else:
        cbar1.set_label('平均相似度')
    
    # ========== 图4: 各类型×维度的中位数相似度热力图（按中位数从高到低排序）==========
    # 计算每个关系类型在所有维度下的中位数的平均值，用于排序
    type_median_avg = {}
    for rel_type in relation_types:
        median_values = [median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        type_median_avg[rel_type] = np.mean(median_values)
    
    # 按中位数从高到低排序
    type_order_by_median = sorted(type_median_avg.keys(), key=lambda x: type_median_avg[x], reverse=True)
    
    # 构建热力图数据矩阵
    heatmap_data_median = []
    for rel_type in type_order_by_median:
        row = [median_data[rel_type].get(dim, 0) for dim in DIMENSIONS_TO_TEST]
        heatmap_data_median.append(row)
    heatmap_data_median = np.array(heatmap_data_median)
    
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
    else:
        cbar2.set_label('中位数相似度')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 4张分析图表已生成并保存为: {output_file}")
    print("\n图表说明:")
    print("  • 图1: 各维度下关系类型排序趋势（使用中位数，按512维中位数排序）")
    print("  • 图2: 各关系类型在不同维度下的中位数变化趋势")
    print("  • 图3: 各类型×维度的平均相似度热力图（按均值从高到低排序）")
    print("  • 图4: 各类型×维度的中位数相似度热力图（按中位数从高到低排序）")

def main():
    print("="*70)
    print("中位数分析图表生成脚本")
    print("="*70)
    
    # 加载数据
    print(f"\n正在加载数据文件: {SUMMARY_JSON_FILE}")
    summary = load_summary_data(SUMMARY_JSON_FILE)
    print(f"✓ 数据加载成功")
    print(f"  测试维度: {summary['测试的维度']}")
    
    # 生成图表
    generate_analysis_charts(summary, OUTPUT_FILE)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)

if __name__ == "__main__":
    main()

