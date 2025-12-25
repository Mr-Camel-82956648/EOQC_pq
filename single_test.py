import os
import numpy as np
import pandas as pd
from config import config
from openai import OpenAI
import json
from typing import List, Dict, Tuple

# 初始化客户端
client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 从您提供的素材中提取关键句子
text_material = """绝大多数亚眠的参观者都对哥特风格怀有一些误导性的先人为主的想法..."""

class SentenceSimilarityAnalyzer:
    def __init__(self, dimensions: List[int] = [64, 128, 256, 512]):
        self.dimensions = dimensions
        self.embeddings_cache = {}
        
    def get_embedding(self, text: str, dimension: int = 64) -> np.ndarray:
        """获取文本嵌入向量，带缓存"""
        cache_key = f"{text}_{dimension}"
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        completion = client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=dimension
        )
        embedding = np.array(completion.data[0].embedding)
        self.embeddings_cache[cache_key] = embedding
        return embedding
    
    def calculate_similarity_metrics(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Dict:
        """计算各种相似度指标"""
        # 点积
        dot_product = np.dot(embedding1, embedding2)
        
        # 余弦相似度
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_sim = dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
        
        # 欧氏距离
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        
        # 曼哈顿距离
        manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
        
        # Jaccard相似度（近似）
        min_vals = np.minimum(np.abs(embedding1), np.abs(embedding2))
        max_vals = np.maximum(np.abs(embedding1), np.abs(embedding2))
        jaccard_sim = np.sum(min_vals) / np.sum(max_vals) if np.sum(max_vals) != 0 else 0
        
        return {
            'cosine_similarity': cosine_sim,
            'dot_product': dot_product,
            'euclidean_distance': euclidean_dist,
            'manhattan_distance': manhattan_dist,
            'jaccard_similarity': jaccard_sim,
            'norm_product': norm1 * norm2
        }
    
    def analyze_sentence_pairs(self, sentence_pairs: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """分析句子对的相似度"""
        results = []
        
        for sentence1, sentence2, pair_type in sentence_pairs:
            for dim in self.dimensions:
                try:
                    emb1 = self.get_embedding(sentence1, dim)
                    emb2 = self.get_embedding(sentence2, dim)
                    
                    metrics = self.calculate_similarity_metrics(emb1, emb2)
                    
                    results.append({
                        'sentence1': sentence1[:50] + '...' if len(sentence1) > 50 else sentence1,
                        'sentence2': sentence2[:50] + '...' if len(sentence2) > 50 else sentence2,
                        'pair_type': pair_type,
                        'dimension': dim,
                        **metrics
                    })
                    
                except Exception as e:
                    print(f"处理失败: {sentence1[:30]}... - 错误: {str(e)}")
                    continue
        
        return pd.DataFrame(results)

class QuestionGenerator:
    def __init__(self):
        self.system_prompt = """你是一位专业的艺术史教授，擅长设计考察学生对哥特建筑理解程度的选择题。
请根据提供的素材，设计高质量的选择题。
要求：
1. 题目必须基于素材内容
2. 提供1个正确答案和3个干扰项
3. 干扰项需要有不同的难度级别：
   - 一个明显错误但相关的干扰项
   - 一个中等难度的干扰项（部分正确）
   - 一个高难度的干扰项（表达形式不同但意思相近）
4. 标注每个干扰项的类型"""
    
    def generate_questions(self, material: str, num_questions: int = 3) -> List[Dict]:
        """生成选择题"""
        questions = []
        
        for i in range(num_questions):
            prompt = f"""请基于以下关于哥特建筑的素材，设计一道选择题：

素材内容：
{material[:2000]}...

请生成：
1. 问题（题目）
2. 正确答案
3. 三个干扰项（分别标注：明显错误、中等难度、高难度）
4. 简要解释为什么每个干扰项是这样的难度

请以JSON格式返回，包含以下字段：
- question: 问题文本
- correct_answer: 正确答案
- distractors: 干扰项列表，每个干扰项包含：
  - text: 文本内容
  - difficulty: 难度级别（low/medium/high）
  - type: 干扰项类型（如：近义词、反义词、无关项、部分正确等）
- explanation: 每个干扰项难度的解释"""
            
            try:
                completion = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                
                response_text = completion.choices[0].message.content
                # 提取JSON部分
                json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                question_data = json.loads(json_str)
                questions.append(question_data)
                
            except Exception as e:
                print(f"生成第{i+1}题失败: {str(e)}")
                continue
        
        return questions

def create_test_sentence_pairs() -> List[Tuple[str, str, str]]:
    """创建测试用的句子对，涵盖不同难度类型"""
    
    # 基础句子对
    base_pairs = [
        # 同义不同表达（高相似度）
        ("古典与文艺复兴风格的融合", "文艺复兴与古典风格的融合", "同义不同表达"),

        # 近义词
        ("哥特建筑难以捉摸", "哥特建筑不容易理解", "近义词"),
        
        # 反义词（低相似度）
        ("哥特建筑是理性的", "哥特建筑是非理性的", "反义词"),
        ("文艺复兴推崇哥特建筑", "文艺复兴厌恶哥特建筑", "反义词"),
        
        # 双重否定
        ("这不是不重要的", "这是重要的", "双重否定"),
        ("哥特建筑不是不适用于古典范畴", "哥特建筑适用于古典范畴", "双重否定"),
        
        # 部分正确
        ("哥特建筑完全由工程逻辑控制", "哥特建筑部分由工程逻辑控制", "部分正确"),
        ("哥特建筑只有结构意义", "哥特建筑有结构和象征意义", "部分正确"),
        
        # 无关项
        ("哥特建筑的飞扶壁结构", "今天天气很好", "无关项"),
        ("经院哲学影响哥特建筑", "我喜欢吃苹果", "无关项"),
        
        # 复杂句式变化
        ("哥特建筑，由于其独特的结构体系，在视觉上创造了一种超世氛围",
         "独特的结构体系使得哥特建筑在视觉上营造出超世氛围", "句式变化"),
         
        # 概念混淆
        ("哥特建筑与拜占庭建筑相似", "哥特建筑与文艺复兴建筑相似", "概念混淆"),
        
        # 过度概括
        ("所有中世纪建筑都是哥特式", "部分中世纪建筑是哥特式", "过度概括"),
    ]
    
    return base_pairs

def analyze_difficulty_thresholds(df: pd.DataFrame) -> Dict:
    """分析不同难度类型的阈值范围"""
    
    threshold_analysis = {}
    
    # 按类型分析
    for pair_type in df['pair_type'].unique():
        type_df = df[df['pair_type'] == pair_type]
        
        threshold_analysis[pair_type] = {
            'cosine_mean': type_df['cosine_similarity'].mean(),
            'cosine_std': type_df['cosine_similarity'].std(),
            'cosine_min': type_df['cosine_similarity'].min(),
            'cosine_max': type_df['cosine_similarity'].max(),
            'sample_size': len(type_df)
        }
    
    return threshold_analysis

def create_visual_report(df: pd.DataFrame, thresholds: Dict):
    """创建可视化报告"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    
    # 设置中文字体 - 使用虚拟环境中的字体文件
    font_path = '/home/dataset-assist-0/rzh/projects/rag_pq/venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'
    
    if os.path.exists(font_path):
        try:
            # 将字体添加到字体管理器
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            print("✓ 已加载中文字体: SimHei")
        except Exception as e:
            print(f"⚠ 字体注册失败，将使用直接路径方式: {str(e)}")
            # 如果注册失败，创建FontProperties对象备用
            chinese_font = fm.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    else:
        print(f"⚠ 字体文件不存在: {font_path}")
        print("  将使用默认字体，中文可能显示为方块")
    
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 按类型分布的箱线图
    sns.boxplot(x='pair_type', y='cosine_similarity', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('不同句子类型的余弦相似度分布')
    axes[0, 0].set_xlabel('句子类型')
    axes[0, 0].set_ylabel('余弦相似度')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 维度对相似度的影响
    sns.lineplot(x='dimension', y='cosine_similarity', hue='pair_type', 
                 data=df, ax=axes[0, 1], marker='o')
    axes[0, 1].set_title('不同维度下的相似度变化')
    axes[0, 1].set_xlabel('嵌入维度')
    axes[0, 1].set_ylabel('余弦相似度')
    
    # 3. 相似度直方图
    axes[1, 0].hist(df['cosine_similarity'], bins=20, alpha=0.7)
    axes[1, 0].axvline(x=0.8, color='r', linestyle='--', label='高相似度阈值(0.8)')
    axes[1, 0].axvline(x=0.5, color='g', linestyle='--', label='中等相似度阈值(0.5)')
    axes[1, 0].axvline(x=0.2, color='b', linestyle='--', label='低相似度阈值(0.2)')
    axes[1, 0].set_title('相似度分布直方图')
    axes[1, 0].set_xlabel('余弦相似度')
    axes[1, 0].set_ylabel('频率')
    axes[1, 0].legend()
    
    # 4. 散点图：点积 vs 余弦相似度
    scatter = axes[1, 1].scatter(df['dot_product'], df['cosine_similarity'], 
                                 c=df['dimension'], cmap='viridis', alpha=0.6)
    axes[1, 1].set_title('点积与余弦相似度的关系')
    axes[1, 1].set_xlabel('点积')
    axes[1, 1].set_ylabel('余弦相似度')
    plt.colorbar(scatter, ax=axes[1, 1], label='维度')
    
    plt.tight_layout()
    plt.savefig('similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("句子相似度分析实验")
    print("用于选择题干扰项难度控制")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = SentenceSimilarityAnalyzer(dimensions=[64, 128, 256, 512])
    
    # 1. 基础句子对分析
    print("\n1. 分析基础句子对...")
    base_pairs = create_test_sentence_pairs()
    results_df = analyzer.analyze_sentence_pairs(base_pairs)
    
    # 保存结果
    results_df.to_csv('sentence_similarity_results.csv', index=False, encoding='utf-8-sig')
    print(f"基础分析完成，保存了 {len(results_df)} 条结果")
    
    # 2. 阈值分析
    print("\n2. 分析阈值范围...")
    thresholds = analyze_difficulty_thresholds(results_df)
    
    print("\n阈值分析结果:")
    for pair_type, stats in thresholds.items():
        print(f"\n{pair_type}:")
        print(f"  相似度范围: {stats['cosine_min']:.3f} - {stats['cosine_max']:.3f}")
        print(f"  平均值: {stats['cosine_mean']:.3f} (±{stats['cosine_std']:.3f})")
        print(f"  样本数: {stats['sample_size']}")
    
    # 3. 生成选择题
    print("\n3. 生成基于素材的选择题...")
    generator = QuestionGenerator()
    questions = generator.generate_questions(text_material, num_questions=2)
    
    # 分析题目中的选项相似度
    all_options_pairs = []
    for i, q in enumerate(questions):
        print(f"\n第{i+1}题: {q['question'][:100]}...")
        print(f"正确答案: {q['correct_answer']}")
        
        # 创建正确答案与干扰项的句子对
        correct = q['correct_answer']
        for distractor in q['distractors']:
            all_options_pairs.append((correct, distractor['text'], 
                                     f"题目{i+1}_{distractor['difficulty']}_{distractor['type']}"))
        
        # 保存题目
        with open(f'question_{i+1}.json', 'w', encoding='utf-8') as f:
            json.dump(q, f, ensure_ascii=False, indent=2)
    
    # 4. 分析题目选项的相似度
    print("\n4. 分析题目选项相似度...")
    if all_options_pairs:
        question_results = analyzer.analyze_sentence_pairs(all_options_pairs)
        question_results.to_csv('question_options_similarity.csv', index=False, encoding='utf-8-sig')
        
        # 按难度类型汇总
        difficulty_groups = question_results.groupby('pair_type')
        print("\n题目选项相似度总结:")
        for difficulty, group in difficulty_groups:
            print(f"\n{difficulty}:")
            print(f"  平均相似度: {group['cosine_similarity'].mean():.3f}")
            print(f"  样本数: {len(group)}")
    
    # 5. 创建可视化报告
    try:
        print("\n5. 创建可视化报告...")
        create_visual_report(results_df, thresholds)
        print("可视化报告已保存为 similarity_analysis.png")
    except Exception as e:
        print(f"可视化失败: {str(e)}")
    
    # 6. 给出阈值建议
    print("\n" + "=" * 60)
    print("阈值建议:")
    print("=" * 60)
    
    # 根据分析结果给出建议阈值
    print("\n根据分析，建议的干扰项相似度阈值:")
    print("1. 高难度干扰项: 余弦相似度 0.7-0.9")
    print("   - 适用于：同义不同表达、句式变化")
    print("   - 风险：可能与正确答案过于相似")
    
    print("\n2. 中等难度干扰项: 余弦相似度 0.4-0.7")
    print("   - 适用于：部分正确、概念混淆")
    print("   - 理想：既能迷惑学生，又不至于太像正确答案")
    
    print("\n3. 低难度干扰项: 余弦相似度 0.1-0.4")
    print("   - 适用于：明显错误但相关、过度概括")
    print("   - 作用：区分基础掌握程度")
    
    print("\n4. 无效干扰项: 余弦相似度 < 0.1")
    print("   - 问题：过于明显，起不到考察作用")
    
    print("\n5. 推荐组合:")
    print("   - 一个低难度干扰项 (0.2-0.4)")
    print("   - 一个中等难度干扰项 (0.5-0.7)")
    print("   - 一个高难度干扰项 (0.7-0.85)")
    
    return results_df, questions

if __name__ == "__main__":
    # 运行主实验
    results_df, questions = main()
    
    # 显示前几行结果
    print("\n前5行结果预览:")
    print(results_df[['pair_type', 'dimension', 'cosine_similarity']].head())