# main.py (简化版本，直接测试)
import json
import time
from data_processor import DataProcessor
from question_generator import QuestionGenerator, Question
from config import config

def test_qwen_api():
    """测试千问API连接"""
    from openai import OpenAI
    
    print("测试千问API连接...")
    
    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好，请简单介绍一下自己"}
            ],
            max_tokens=50
        )
        print(f"✅ API连接成功！")
        print(f"响应: {response.choices[0].message.content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False

def run_single_experiment():
    """运行单次实验"""
    print("=" * 60)
    print("实验3.2：选项质量与分布特征分析（单次测试）")
    print("=" * 60)
    
    # 0. 测试API连接
    if not test_qwen_api():
        print("请检查DASHSCOPE_API_KEY配置")
        return
    
    # 1. 数据准备
    print("\n1. 准备数据...")
    processor = DataProcessor()
    sample_text = processor.get_sample_text()
    chunks = processor.split_into_chunks(sample_text, config.NUM_QUESTIONS)
    
    print(f"文本长度: {len(sample_text)} 字符")
    print(f"分割成 {len(chunks)} 个片段")
    print(f"第一个片段预览:\n{chunks[0][:200]}...\n")
    
    # 2. 初始化生成器
    generator = QuestionGenerator()
    
    # 3. 生成Baseline题目
    print("\n" + "-" * 40)
    print("Baseline方法")
    print("-" * 40)
    baseline_question = generator.generate_baseline_question(chunks[0])
    
    # 为Baseline计算相似度
    baseline_similarities = generator.calculate_baseline_similarities(baseline_question)
    
    # 4. 生成RAG题目
    print("\n" + "-" * 40)
    print("RAG方法")
    print("-" * 40)
    rag_question = generator.generate_rag_question(chunks[0])
    
    # 5. 显示结果
    print("\n" + "=" * 60)
    print("实验结果对比")
    print("=" * 60)
    
    # Baseline结果
    print("\n📊 Baseline方法结果:")
    print(f"问题: {baseline_question.text}")
    print("\n选项:")
    for i, option in enumerate(baseline_question.options):
        correct_mark = "✓" if i == baseline_question.correct_idx else " "
        similarity = baseline_similarities[i] if baseline_similarities else 0
        print(f"  {correct_mark} {i+1}. {option} (相似度: {similarity:.3f})")
    
    print(f"\n⏱️ 时间成本:")
    print(f"  生成时间: {baseline_question.generation_time:.3f}s")
    print(f"  Embedding时间: {baseline_question.embedding_time:.3f}s")
    print(f"  总时间: {baseline_question.generation_time + baseline_question.embedding_time:.3f}s")
    
    # RAG结果 - 显示原始8个选项
    print("\n📊 RAG方法结果:")
    print(f"问题: {rag_question.text}")
    print("\n原始8个选项及其相似度:")
    for i, option in enumerate(rag_question.original_options):
        correct_mark = "✓" if i == rag_question.original_correct_idx else " "
        similarity = rag_question.original_similarities[i] if rag_question.original_similarities else 0
        print(f"  {correct_mark} {i+1}. {option} (相似度: {similarity:.3f})")

    # 显示排序和过滤过程
    print("\n🔍 选项过滤过程:")
    # 获取干扰项的索引和相似度
    distractor_indices = [i for i in range(len(rag_question.original_options)) 
                        if i != rag_question.original_correct_idx]
    distractor_similarities = [rag_question.original_similarities[i] for i in distractor_indices]

    # 按相似度对干扰项进行排序
    sorted_distractors = sorted(
        zip(distractor_indices, distractor_similarities),
        key=lambda x: x[1]
    )

    print("  干扰项按相似度排序:")
    for rank, (original_idx, similarity) in enumerate(sorted_distractors):
        option_text = rag_question.original_options[original_idx]
        
        # 判断是否被删除
        if len(sorted_distractors) >= 5:
            if rank < 2:  # 两个最不相似的
                tag = "❌ 删除 (最不相似)"
            elif rank >= len(sorted_distractors) - 2:  # 两个最相似的
                tag = "❌ 删除 (最相似)"
            else:
                tag = "✅ 保留"
        else:
            # 如果干扰项少于5个，使用简化的过滤逻辑
            keep_count = max(0, len(sorted_distractors) - 4)
            remove_each_side = (len(sorted_distractors) - keep_count) // 2
            if rank < remove_each_side or rank >= len(sorted_distractors) - remove_each_side:
                tag = "❌ 删除"
            else:
                tag = "✅ 保留"
        
        print(f"    相似度 {similarity:.3f}: {option_text} - {tag}")

    # 显示哪些干扰项被选中（从最终选项中获取）
    print(f"\n📦 最终过滤后的4个选项:")
    for i, option in enumerate(rag_question.options):
        correct_mark = "✓" if i == rag_question.correct_idx else " "
        similarity = rag_question.similarities[i] if rag_question.similarities else 0
        print(f"  {correct_mark} {i+1}. {option} (相似度: {similarity:.3f})")

    print(f"\n⏱️ 时间成本:")
    print(f"  生成时间: {rag_question.generation_time:.3f}s")
    print(f"  Embedding时间: {rag_question.embedding_time:.3f}s")
    print(f"  过滤时间: {rag_question.filtering_time:.3f}s")
    print(f"  总时间: {rag_question.generation_time + rag_question.embedding_time + rag_question.filtering_time:.3f}s")
    
    # 6. 统计分析
    print("\n" + "=" * 60)
    print("统计分析")
    print("=" * 60)
    
    # 提取干扰项相似度（排除正确选项）
    baseline_distractor_sims = []
    if baseline_similarities:
        baseline_distractor_sims = [sim for i, sim in enumerate(baseline_similarities) 
                                   if i != baseline_question.correct_idx]
    
    rag_distractor_sims = []
    if rag_question.similarities:
        rag_distractor_sims = [sim for i, sim in enumerate(rag_question.similarities) 
                              if i != rag_question.correct_idx]
    
    # 计算统计指标
    import numpy as np
    
    if baseline_distractor_sims:
        print(f"\nBaseline干扰项相似度统计:")
        print(f"  平均值: {np.mean(baseline_distractor_sims):.3f}")
        print(f"  标准差: {np.std(baseline_distractor_sims):.3f}")
        print(f"  范围: [{min(baseline_distractor_sims):.3f}, {max(baseline_distractor_sims):.3f}]")
    
    if rag_distractor_sims:
        print(f"\nRAG干扰项相似度统计:")
        print(f"  平均值: {np.mean(rag_distractor_sims):.3f}")
        print(f"  标准差: {np.std(rag_distractor_sims):.3f}")
        print(f"  范围: [{min(rag_distractor_sims):.3f}, {max(rag_distractor_sims):.3f}]")
    
    # 长度偏差分析
    baseline_lengths = [len(opt) for opt in baseline_question.options]
    rag_lengths = [len(opt) for opt in rag_question.options]
    
    baseline_correct_length = baseline_lengths[baseline_question.correct_idx]
    rag_correct_length = rag_lengths[rag_question.correct_idx]
    
    print(f"\n长度偏差分析:")
    print(f"  Baseline正确选项长度: {baseline_correct_length}")
    print(f"  Baseline选项长度范围: [{min(baseline_lengths)}, {max(baseline_lengths)}]")
    print(f"  正确选项是否最长或最短: {baseline_correct_length == max(baseline_lengths) or baseline_correct_length == min(baseline_lengths)}")
    
    print(f"\n  RAG正确选项长度: {rag_correct_length}")
    print(f"  RAG选项长度范围: [{min(rag_lengths)}, {max(rag_lengths)}]")
    print(f"  正确选项是否最长或最短: {rag_correct_length == max(rag_lengths) or rag_correct_length == min(rag_lengths)}")
    
    # 7. 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    results = {
        "experiment_info": {
            "experiment_name": "3.2 选项质量与分布特征分析（单次测试）",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.GENERATION_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dimensions": config.EMBEDDING_DIMENSIONS,
            "num_questions": config.NUM_QUESTIONS
        },
        "baseline": baseline_question.to_dict(),
        "rag": rag_question.to_dict(),
        "statistics": {
            "baseline": {
                "distractor_similarity_mean": float(np.mean(baseline_distractor_sims)) if baseline_distractor_sims else 0,
                "distractor_similarity_std": float(np.std(baseline_distractor_sims)) if baseline_distractor_sims else 0,
                "length_bias": baseline_correct_length == max(baseline_lengths) or baseline_correct_length == min(baseline_lengths)
            },
            "rag": {
                "distractor_similarity_mean": float(np.mean(rag_distractor_sims)) if rag_distractor_sims else 0,
                "distractor_similarity_std": float(np.std(rag_distractor_sims)) if rag_distractor_sims else 0,
                "length_bias": rag_correct_length == max(rag_lengths) or rag_correct_length == min(rag_lengths)
            }
        }
    }
    
    # 保存到文件
    with open("single_experiment_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: single_experiment_result.json")
    
    # 8. 总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    
    print(f"\n✅ 实验完成！")
    
    if baseline_question.generation_time + baseline_question.embedding_time > 0:
        time_ratio = (rag_question.generation_time + rag_question.embedding_time + rag_question.filtering_time) / \
                     (baseline_question.generation_time + baseline_question.embedding_time)
        print(f"总耗时:")
        print(f"  Baseline: {baseline_question.generation_time + baseline_question.embedding_time:.2f}s")
        print(f"  RAG: {rag_question.generation_time + rag_question.embedding_time + rag_question.filtering_time:.2f}s")
        print(f"  RAG比Baseline慢 {time_ratio:.1f} 倍")
    else:
        print("无法计算时间比率")
    
    return results

if __name__ == "__main__":
    # 检查API密钥
    from config import config
    
    if not config.DASHSCOPE_API_KEY:
        print("❌ 错误: 未找到DASHSCOPE_API_KEY环境变量")
        print("请在.env文件中设置DASHSCOPE_API_KEY=your_api_key")
        exit(1)
    
    print("✅ 环境检查完成")
    print(f"📊 当前嵌入维度: {config.EMBEDDING_DIMENSIONS}")
    print(f"📊 嵌入模型: {config.EMBEDDING_MODEL}")
    run_single_experiment()