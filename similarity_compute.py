import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from openai import OpenAI
from config import config
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

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
EMBEDDING_MODEL = "Qwen3-Embedding-4B"  # Qwen3-Embedding-8B  Qwen3-Embedding-4B
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


def main():
    # 创建输出目录 result/{model_name}/
    model_name = EMBEDDING_MODEL.replace('/', '_').replace(' ', '_')
    output_dir = os.path.join('result', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据模式确定输出文件名
    if SAMPLE_RATIO < 1.0:
        log_file = os.path.join(output_dir, f'similarity_analysis_test_{int(SAMPLE_RATIO*100)}pct.log')
        results_file = os.path.join(output_dir, RESULTS_CSV_FILE.replace('.csv', f'_test_{int(SAMPLE_RATIO*100)}pct.csv'))
        summary_file = os.path.join(output_dir, SUMMARY_JSON_FILE.replace('.json', f'_test_{int(SAMPLE_RATIO*100)}pct.json'))
    else:
        log_file = os.path.join(output_dir, 'similarity_analysis.log')
        results_file = os.path.join(output_dir, RESULTS_CSV_FILE)
        summary_file = os.path.join(output_dir, SUMMARY_JSON_FILE)
    
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
        
        # 7. 保存统计摘要 (JSON)
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
        print(f"• 日志文件: {log_file}")
        print(f"\n[提示] 可视化图表请使用 similarity_visualize.py 脚本生成")
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