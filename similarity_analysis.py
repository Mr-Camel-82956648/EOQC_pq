import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from openai import OpenAI
from config import config
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# ========== 中文字体配置 ==========
def setup_chinese_font():
    """设置中文字体，解决图表中文显示问题"""
    font_path = '/home/dataset-assist-0/rzh/projects/rag_pq/venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'
    
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

# ========== 日志输出类 ==========
class TeeOutput:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        if self.file:
            self.file.close()

# ========== 模型和API配置 ==========
client = OpenAI(
    api_key=config.GITEEAI_API_KEY,
    base_url="https://ai.gitee.com/v1"
)
EMBEDDING_MODEL = "Qwen3-Embedding-8B"
DIMENSIONS_TO_TEST = [32, 64, 128, 256, 512]
SENTENCE_PAIRS_FILE = "sentence_pairs_1800.json"
RESULTS_CSV_FILE = "similarity_results_large_scale.csv"
SUMMARY_JSON_FILE = "similarity_summary_large_scale.json"

# ========== 测试模式配置 ==========
SAMPLE_RATIO = 1.0  # 小规模测试：0.05，完整运行：1.0

class SentenceSimilarityAnalyzer:
    def __init__(self, dimensions: List[int] = DIMENSIONS_TO_TEST):
        self.dimensions = dimensions
        self.embeddings_cache = {}

    def get_embedding(self, text: str, dimension: int) -> np.ndarray:
        cache_key = (text, dimension)
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=dimension
            )
            embedding = np.array(response.data[0].embedding)
            self.embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"[错误] 获取嵌入失败: '{text[:30]}...', 维度: {dimension}, 错误: {e}")
            return np.zeros(dimension)

    def calculate_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 * norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def analyze_sentence_pairs(self, sentence_pairs: List[Tuple[str, str, str]]) -> pd.DataFrame:
        all_results = []
        total_pairs = len(sentence_pairs)
        print(f"开始分析 {total_pairs} 对句子，共 {len(self.dimensions)} 种维度...")
        
        for idx, (sent1, sent2, rel_type) in enumerate(sentence_pairs):
            if idx % 10 == 0:
                print(f"  进度: {idx}/{total_pairs}")
            
            for dim in self.dimensions:
                emb1 = self.get_embedding(sent1, dim)
                emb2 = self.get_embedding(sent2, dim)
                cosine_sim = self.calculate_cosine_similarity(emb1, emb2)
                
                all_results.append({
                    '句子1': sent1,
                    '句子2': sent2,
                    '关系类型': rel_type,
                    '嵌入维度': dim,
                    '余弦相似度': cosine_sim,
                })
        
        return pd.DataFrame(all_results)

def sample_balanced_by_relation_type(sentence_pairs: List[Tuple[str, str, str]], 
                                     sample_ratio: float, 
                                     seed: int = 42) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    grouped_by_type = defaultdict(list)
    for sent1, sent2, rel_type in sentence_pairs:
        grouped_by_type[rel_type].append((sent1, sent2, rel_type))
    
    total_pairs = len(sentence_pairs)
    num_types = len(grouped_by_type)
    target_total = max(1, int(total_pairs * sample_ratio))
    samples_per_type = max(1, target_total // num_types)
    
    random.seed(seed)
    sampled_pairs = []
    type_counts = {}
    
    for rel_type, pairs in grouped_by_type.items():
        available_count = len(pairs)
        actual_sample_size = min(samples_per_type, available_count)
        sampled = random.sample(pairs, actual_sample_size)
        sampled_pairs.extend(sampled)
        type_counts[rel_type] = actual_sample_size
    
    random.shuffle(sampled_pairs)
    return sampled_pairs, type_counts

def load_sentence_pairs(filepath: str) -> Optional[List[Tuple[str, str, str]]]:
    if not os.path.exists(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("[错误] JSON文件格式不正确，根元素应为列表。")
            return None
        
        sentence_pairs = []
        relation_counts = {}
        for item in data:
            sent1 = item.get('sentence1', '').strip()
            sent2 = item.get('sentence2', '').strip()
            rel_type = item.get('relation_type', '').strip()
            
            if sent1 and sent2 and rel_type:
                # 将英文标签映射为中文，便于后续阅读
                chinese_labels = {
                    'synonym_different_expression': '同义不同表达',
                    'synonym': '近义词',
                    'antonym': '反义词',
                    'double_negation': '双重否定',
                    'partially_correct': '部分正确',
                    'irrelevant': '无关项',
                    'sentence_structure': '句式变化',
                    'concept_confusion': '概念混淆',
                    'overgeneralization': '过度概括'
                }
                chinese_rel_type = chinese_labels.get(rel_type, rel_type)
                sentence_pairs.append((sent1, sent2, chinese_rel_type))
                relation_counts[chinese_rel_type] = relation_counts.get(chinese_rel_type, 0) + 1
        
        print(f"成功加载 {len(sentence_pairs)} 对句子。")
        print("关系类型分布:")
        for rel_type, count in sorted(relation_counts.items()):
            print(f"  {rel_type}: {count} 对")
        
        return sentence_pairs
    except Exception as e:
        print(f"[错误] 加载JSON文件失败: {e}")
        return None

# ========== 增强的统计分析函数 ==========
def create_detailed_statistical_summary(df: pd.DataFrame) -> Dict:
    """创建详细的统计摘要，按维度分别统计"""
    summary = {
        '测试的维度': DIMENSIONS_TO_TEST,
        '按维度的统计': {},
        '按关系类型的整体统计': {}
    }
    
    # 1. 按维度统计
    print("\n" + "="*70)
    print("详细统计分析结果 (按维度)")
    print("="*70)
    
    for dim in DIMENSIONS_TO_TEST:
        dim_df = df[df['嵌入维度'] == dim]
        if dim_df.empty:
            continue
        
        summary['按维度的统计'][dim] = {}
        dim_stats = summary['按维度的统计'][dim]
        
        # 按关系类型统计该维度下的数据
        for rel_type in sorted(dim_df['关系类型'].unique()):
            type_df = dim_df[dim_df['关系类型'] == rel_type]
            sim_values = type_df['余弦相似度']
            
            dim_stats[rel_type] = {
                '样本数': len(type_df),
                '平均值': float(sim_values.mean()),
                '标准差': float(sim_values.std()),
                '最小值': float(sim_values.min()),
                '25%分位数': float(sim_values.quantile(0.25)),
                '中位数': float(sim_values.quantile(0.50)),
                '75%分位数': float(sim_values.quantile(0.75)),
                '最大值': float(sim_values.max())
            }
        
        # 打印该维度的统计表格
        print(f"\n维度 {dim}:")
        print("-" * 95)
        print(f"{'关系类型':<12} {'样本数':<6} {'平均值':<8} {'标准差':<8} {'范围':<22} {'中位数':<8}")
        print("-" * 95)
        
        for rel_type in sorted(dim_stats.keys()):
            stats = dim_stats[rel_type]
            range_str = f"[{stats['最小值']:.3f}, {stats['最大值']:.3f}]"
            print(f"{rel_type:<12} {stats['样本数']:<6} {stats['平均值']:.4f}    {stats['标准差']:.4f}    {range_str:<22} {stats['中位数']:.4f}")
    
    # 2. 综合所有维度的关系类型统计（用于阈值建议）
    print("\n" + "="*70)
    print("综合统计 (所有维度)")
    print("="*70)
    
    for rel_type in sorted(df['关系类型'].unique()):
        type_df = df[df['关系类型'] == rel_type]
        sim_values = type_df['余弦相似度']
        
        summary['按关系类型的整体统计'][rel_type] = {
            '总样本数': len(type_df),
            '平均值': float(sim_values.mean()),
            '标准差': float(sim_values.std()),
            '最小值': float(sim_values.min()),
            '5%分位数': float(sim_values.quantile(0.05)),
            '25%分位数': float(sim_values.quantile(0.25)),
            '中位数': float(sim_values.quantile(0.50)),
            '75%分位数': float(sim_values.quantile(0.75)),
            '95%分位数': float(sim_values.quantile(0.95)),
            '最大值': float(sim_values.max())
        }
    
    # 打印综合统计表
    print(f"\n{'关系类型':<12} {'总样本':<6} {'平均值':<8} {'标准差':<8} {'5%-95%区间':<22} {'中位数':<8}")
    print("-" * 95)
    
    for rel_type in sorted(summary['按关系类型的整体统计'].keys()):
        stats = summary['按关系类型的整体统计'][rel_type]
        interval_str = f"[{stats['5%分位数']:.3f}, {stats['95%分位数']:.3f}]"
        print(f"{rel_type:<12} {stats['总样本数']:<6} {stats['平均值']:.4f}    {stats['标准差']:.4f}    {interval_str:<22} {stats['中位数']:.4f}")
    
    return summary

def analyze_dimension_impact(summary: Dict):
    """分析维度对相似度结果的影响"""
    print("\n" + "="*70)
    print("维度影响分析")
    print("="*70)
    
    impact_results = {}
    
    # 对每种关系类型，分析不同维度间的差异
    for rel_type in summary['按关系类型的整体统计'].keys():
        dim_means = []
        dim_stds = []
        
        for dim in DIMENSIONS_TO_TEST:
            if dim in summary['按维度的统计'] and rel_type in summary['按维度的统计'][dim]:
                stats = summary['按维度的统计'][dim][rel_type]
                dim_means.append(stats['平均值'])
                dim_stds.append(stats['标准差'])
        
        if len(dim_means) > 1:
            # 计算不同维度间的变异系数
            mean_of_means = np.mean(dim_means)
            std_of_means = np.std(dim_means)
            cv_of_means = std_of_means / mean_of_means if mean_of_means != 0 else 0
            
            # 计算最大差异
            max_diff = max(dim_means) - min(dim_means)
            
            # 判断趋势
            if dim_means[-1] > dim_means[0] + 0.01:
                trend = '随维度增加而轻微上升'
            elif dim_means[-1] < dim_means[0] - 0.01:
                trend = '随维度增加而轻微下降'
            else:
                trend = '基本保持稳定'
            
            impact_results[rel_type] = {
                '各维度平均相似度均值': mean_of_means,
                '各维度平均相似度标准差': std_of_means,
                '变异系数(CV)': cv_of_means,
                '维度间最大差异': max_diff,
                '变化趋势': trend
            }
    
    # 打印维度影响分析结果
    print(f"\n{'关系类型':<12} {'平均相似度':<12} {'维度间标准差':<12} {'变异系数':<12} {'最大差异':<12} {'趋势':<20}")
    print("-" * 95)
    
    for rel_type, impact in impact_results.items():
        print(f"{rel_type:<12} {impact['各维度平均相似度均值']:.4f}      {impact['各维度平均相似度标准差']:.4f}        {impact['变异系数(CV)']:.4f}        {impact['维度间最大差异']:.4f}        {impact['变化趋势']:<20}")
    
    # 计算所有关系类型的平均变异系数（仅用于数据记录，不输出结论）
    avg_cv = np.mean([impact['变异系数(CV)'] for impact in impact_results.values()])
    print(f"\n平均变异系数: {avg_cv:.4f}")

def generate_visualizations(df: pd.DataFrame, output_file: str = 'similarity_analysis.png'):
    """生成可视化图表"""
    print("\n" + "="*70)
    print("开始生成可视化图表 (共6张)")
    print("="*70)
    
    # 图表详细说明
    print("\n图表说明:")
    print("1. 左上图 (箱线图): 展示不同关系类型的余弦相似度整体分布情况。")
    print("   - 用途: 直观比较各类型相似度的中位数和离散程度，箱体代表25%-75%分位数。")
    print("")
    print("2. 中上图 (箱线图): 展示不同嵌入维度下的余弦相似度分布。")
    print("   - 用途: 观察维度本身是否对相似度计算结果有系统性影响。")
    print("")
    print("3. 右上图 (热力图): 展示每种关系类型在每个维度下的平均相似度。")
    print("   - 用途: 颜色越暖（偏红），平均相似度越高，可直观看出不同类型在不同维度下的表现。")
    print("")
    print("4. 左下图 (折线图): 比较所有维度（32, 64, 128, 256, 512维）下，各关系类型的平均相似度排序趋势。")
    print("   - 用途: 观察不同维度下，各关系类型的相对排序是否一致，判断维度对排序的影响。")
    print("")
    print("5. 中下图 (折线图): 展示各关系类型在不同维度下的平均相似度变化趋势。")
    print("   - 用途: 观察维度变化对每种关系类型相似度的影响，判断是否存在维度饱和点。")
    print("")
    print("6. 右下图 (直方图): 展示全部样本余弦相似度的整体分布。")
    print("   - 用途: 显示相似度集中区域，三条虚线为建议的难度阈值参考线。")
    
    # 如果中文字体不可用，准备英文标签作为备选
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
    
    # 设置全局字体（如果可用）
    if chinese_font_available:
        # 设置全局字体属性，影响所有文本
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
    # 设置tick标签字体
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
    
    if not chinese_font_available:
        axes[0, 2].set_yticklabels(pivot_data.index)
    else:
        axes[0, 2].set_yticklabels(pivot_data.index)
        # 设置y轴标签字体
        for label in axes[0, 2].get_yticklabels():
            label.set_fontproperties(chinese_font_prop)
    
    cbar = plt.colorbar(im, ax=axes[0, 2])
    if chinese_font_available:
        cbar.set_label('平均相似度', fontproperties=chinese_font_prop)
    else:
        cbar.set_label('平均相似度')
    
    # 4. 所有维度下关系类型趋势（包含512维）
    # 为了保持排序一致，使用512维的排序作为基准
    dim_512_df = df[df['嵌入维度'] == 512]
    if not chinese_font_available:
        type_means_512 = dim_512_df.groupby('关系类型_en')['余弦相似度'].mean().sort_values()
        type_order = type_means_512.index.tolist()
    else:
        type_means_512 = dim_512_df.groupby('关系类型')['余弦相似度'].mean().sort_values()
        type_order = type_means_512.index.tolist()
    
    # 为不同维度设置不同的标记和颜色
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    
    for i, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_df = df[df['嵌入维度'] == dim]
        if not chinese_font_available:
            type_means = dim_df.groupby('关系类型_en')['余弦相似度'].mean()
        else:
            type_means = dim_df.groupby('关系类型')['余弦相似度'].mean()
        
        # 按照512维的排序重新排列
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n可视化图表已生成并保存为: {output_file}")
    
    # 生成每个维度的单独箱型图
    print("\n开始生成各维度的单独箱型图...")
    dimension_output_file = output_file.replace('.png', '_by_dimension.png')
    
    # 计算子图布局：5个维度，可以排成2行3列或3行2列
    fig_dim, axes_dim = plt.subplots(2, 3, figsize=(20, 13))
    axes_dim = axes_dim.flatten()  # 展平为1维数组便于索引
    
    for idx, dim in enumerate(DIMENSIONS_TO_TEST):
        dim_df = df[df['嵌入维度'] == dim]
        if dim_df.empty:
            continue
        
        ax = axes_dim[idx]
        
        # 为每个维度生成箱型图
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
        
        # 设置tick标签字体
        if chinese_font_available:
            for label in ax.get_xticklabels():
                label.set_fontproperties(chinese_font_prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(chinese_font_prop)
    
    # 隐藏最后一个空的子图（如果有6个子图但只有5个维度）
    if len(DIMENSIONS_TO_TEST) < 6:
        axes_dim[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(dimension_output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"各维度的单独箱型图已生成并保存为: {dimension_output_file}")

def main():
    # 根据模式确定输出文件名
    if SAMPLE_RATIO < 1.0:
        log_file = f'similarity_analysis_test_{int(SAMPLE_RATIO*100)}pct.log'
        results_file = RESULTS_CSV_FILE.replace('.csv', f'_test_{int(SAMPLE_RATIO*100)}pct.csv')
        summary_file = SUMMARY_JSON_FILE.replace('.json', f'_test_{int(SAMPLE_RATIO*100)}pct.json')
        viz_file = 'similarity_analysis_test.png'
    else:
        log_file = 'similarity_analysis.log'
        results_file = RESULTS_CSV_FILE
        summary_file = SUMMARY_JSON_FILE
        viz_file = 'similarity_analysis.png'
    
    # 设置日志输出（同时输出到控制台和文件）
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print("="*70)
        print(f"分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        if SAMPLE_RATIO < 1.0:
            print(f"小规模测试模式 - 处理 {SAMPLE_RATIO*100:.1f}% 的数据")
        else:
            print("完整模式 - 处理全部数据")
        print("="*70)
        
        # 1. 加载句子对数据
        sentence_pairs = load_sentence_pairs(SENTENCE_PAIRS_FILE)
        if sentence_pairs is None:
            return
        
        # 2. 如果启用测试模式，按关系类型平衡采样
        original_count = len(sentence_pairs)
        if SAMPLE_RATIO < 1.0:
            sentence_pairs, type_counts = sample_balanced_by_relation_type(
                sentence_pairs, SAMPLE_RATIO, seed=42
            )
            print(f"\n[测试模式] 从 {original_count} 对句子中平衡采样 {len(sentence_pairs)} 对 ({SAMPLE_RATIO*100:.1f}%)")
        
        # 3. 初始化分析器并进行分析
        analyzer = SentenceSimilarityAnalyzer()
        results_df = analyzer.analyze_sentence_pairs(sentence_pairs)
        
        if results_df.empty:
            print("[错误] 未能生成分析结果。")
            return
        
        # 4. 保存详细结果 (CSV)
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"\n详细分析结果已保存到 (CSV): {results_file}")
        
        # 5. 生成详细的统计分析
        summary = create_detailed_statistical_summary(results_df)
        
        # 6. 分析维度影响
        analyze_dimension_impact(summary)
        
        # 7. 生成可视化图表
        try:
            generate_visualizations(results_df, viz_file)
        except Exception as e:
            print(f"[警告] 生成可视化图表时出错: {e}")
        
        # 8. 保存统计摘要 (JSON)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n统计摘要已保存到 (JSON): {summary_file}")
        
        print("\n" + "="*70)
        print("分析完成！")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"• 原始数据: {SENTENCE_PAIRS_FILE}")
        print(f"• 详细结果 (CSV): {results_file}")
        print(f"• 统计摘要 (JSON): {summary_file}")
        print(f"• 可视化图表 (PNG): {viz_file}")
        print(f"• 日志文件: {log_file}")
        print(f"• 使用模型: {EMBEDDING_MODEL}")
        print(f"• 测试维度: {DIMENSIONS_TO_TEST}")
        if SAMPLE_RATIO < 1.0:
            print(f"\n[提示] 当前为测试模式 ({SAMPLE_RATIO*100:.1f}%)，如需处理全部数据，请将 SAMPLE_RATIO 设置为 1.0")
        print("="*70)
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = tee.stdout
        tee.close()

if __name__ == "__main__":
    main()