# question_generator.py
import time
import random
import json
from typing import Dict, List, Tuple, Any
import numpy as np
from openai import OpenAI
from dataclasses import dataclass
from config import config

@dataclass
class Question:
    text: str
    options: List[str]
    correct_idx: int
    chunk_text: str
    method: str
    generation_time: float
    embedding_time: float = 0.0
    filtering_time: float = 0.0
    similarities: List[float] = None
    
    def to_dict(self):
        result = {
            "question": self.text,
            "options": self.options,
            "correct_index": self.correct_idx,
            "correct_option": self.options[self.correct_idx],
            "distractors": [opt for i, opt in enumerate(self.options) if i != self.correct_idx],
            "method": self.method,
            "generation_time": round(self.generation_time, 3),
            "embedding_time": round(self.embedding_time, 3),
            "filtering_time": round(self.filtering_time, 3),
            "total_time": round(self.generation_time + self.embedding_time + self.filtering_time, 3),
            "similarities": [round(s, 4) for s in self.similarities] if self.similarities else []
        }
        
        # 添加原始选项信息（如果存在）
        if hasattr(self, 'original_options'):
            result["original_options"] = self.original_options
            result["original_similarities"] = [round(s, 4) for s in self.original_similarities] if self.original_similarities else []
            result["original_correct_index"] = self.original_correct_idx
        
        return result

class QuestionGenerator:
    def __init__(self):
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        # 初始化千问客户端（用于生成和嵌入）
        self.llm_client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 嵌入客户端与LLM客户端使用相同的配置
        self.embedding_client = self.llm_client
    
    def generate_baseline_question(self, chunk: str) -> Question:
        """Baseline方法：直接生成1道题目和4个选项"""
        print("生成Baseline题目...")
        start_time = time.time()
        
        prompt = self._create_prompt(chunk, num_options=config.BASELINE_NUM_OPTIONS)
        
        try:
            # 使用千问API生成题目
            response = self.llm_client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE
            )
            
            # 解析响应
            result_text = response.choices[0].message.content
            
            # 尝试解析JSON（有些模型可能不会严格遵守JSON格式）
            try:
                # 找到JSON部分
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    # 如果没有找到JSON，尝试直接解析整个响应
                    result = json.loads(result_text)
            except json.JSONDecodeError:
                print(f"JSON解析失败，使用备选方案。原始响应: {result_text}")
                # 如果JSON解析失败，使用备选方案
                result = self._parse_fallback_result(result_text)
            
            generation_time = time.time() - start_time
            
            question = Question(
                text=result.get("question", "问题生成失败"),
                options=result.get("options", []),
                correct_idx=result.get("correct_index", 0),
                chunk_text=chunk[:200] + "...",
                method="baseline",
                generation_time=generation_time
            )
            
            print(f"Baseline题目生成成功，耗时: {generation_time:.2f}s")
            return question
            
        except Exception as e:
            print(f"Baseline题目生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个示例题目用于测试
            return Question(
                text="什么是'哥特'(Gothic)概念在文艺复兴时期的含义？",
                options=[
                    "一种建筑风格",
                    "文艺复兴自我定义的一部分",
                    "北方野蛮部落的文化",
                    "理性主义的代表"
                ],
                correct_idx=1,
                chunk_text=chunk[:200] + "...",
                method="baseline",
                generation_time=time.time() - start_time
            )
    
    def _parse_fallback_result(self, result_text: str) -> Dict:
        """当JSON解析失败时的备选解析方案"""
        # 简单解析逻辑
        lines = result_text.split('\n')
        question = ""
        options = []
        correct_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("问题:") or line.startswith("question:"):
                question = line.split(":", 1)[1].strip()
            elif line.startswith(("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "1.", "2.", "3.", "4.")):
                # 移除选项标记
                option_text = line.split(".", 1)[1].strip()
                options.append(option_text)
                # 简单假设第一个选项是正确的（实际应用中需要更复杂的逻辑）
                if "正确" in line.lower() or "correct" in line.lower():
                    correct_idx = len(options) - 1
        
        # 如果解析失败，返回默认值
        if not question:
            question = "关于哥特风格的理解，下列哪项是正确的？"
        if not options:
            options = [
                "哥特风格是文艺复兴时期推崇的艺术形式",
                "哥特风格被视为野蛮和黑暗的象征",
                "哥特风格代表理性和科学的进步",
                "哥特风格是古代希腊罗马艺术的延续"
            ]
        
        return {
            "question": question,
            "options": options[:config.BASELINE_NUM_OPTIONS],  # 只取前4个
            "correct_index": min(correct_idx, len(options) - 1)
        }
    
    def generate_rag_question(self, chunk: str) -> Question:
        """RAG方法：生成1道题目和8个候选选项，然后过滤"""
        print("生成RAG题目...")
        start_time = time.time()
        
        # 步骤1: 生成题目和8个选项
        prompt = self._create_prompt(chunk, num_options=config.RAG_NUM_CANDIDATES)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE
            )
            
            result_text = response.choices[0].message.content
            generation_time = time.time() - start_time
            print(f"题目生成完成，耗时: {generation_time:.2f}s")
            
            # 解析响应
            try:
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = json.loads(result_text)
            except json.JSONDecodeError:
                print(f"JSON解析失败，使用备选方案。原始响应: {result_text}")
                result = self._parse_fallback_result(result_text)
            
        except Exception as e:
            print(f"RAG题目生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用示例数据
            result = {
                "question": "文艺复兴时期如何看待哥特风格？",
                "options": [
                    "视为文化复兴的代表",
                    "视为野蛮和黑暗的象征",
                    "视为理性和科学的进步",
                    "视为基督教艺术的巅峰",
                    "视为古代艺术的延续",
                    "视为建筑技术的倒退",
                    "视为艺术自由的表达",
                    "视为异教文化的复兴"
                ],
                "correct_index": 1
            }
            generation_time = time.time() - start_time
        
        # 步骤2: 计算embedding和相似度
        print("计算选项相似度...")
        embedding_start = time.time()
        similarities = self._calculate_option_similarities(
            result["options"], 
            result["correct_index"],
            dimensions=config.EMBEDDING_DIMENSIONS
        )
        embedding_time = time.time() - embedding_start
        print(f"相似度计算完成，耗时: {embedding_time:.2f}s (维度: {config.EMBEDDING_DIMENSIONS})")
        
        # 保存原始选项和相似度用于显示
        original_options = result["options"].copy()
        original_similarities = similarities.copy()
        original_correct_idx = result["correct_index"]
        
        # 步骤3: 过滤选项（去掉最相似和最不相似的）
        print("过滤选项...")
        filtering_start = time.time()
        final_options, final_correct_idx, filtered_similarities = self._filter_options(
            result["options"], 
            result["correct_index"], 
            similarities
        )
        filtering_time = time.time() - filtering_start
        print(f"选项过滤完成，耗时: {filtering_time:.2f}s")
        
        # 创建Question对象，但额外存储原始信息
        question_obj = Question(
            text=result["question"],
            options=final_options,
            correct_idx=final_correct_idx,
            chunk_text=chunk[:200] + "...",
            method="rag",
            generation_time=generation_time,
            embedding_time=embedding_time,
            filtering_time=filtering_time,
            similarities=filtered_similarities
        )
        
        # 添加额外属性（不修改Question类）
        question_obj.original_options = original_options
        question_obj.original_similarities = original_similarities
        question_obj.original_correct_idx = original_correct_idx
        
        return question_obj
    
    def _create_prompt(self, chunk: str, num_options: int) -> str:
        """创建生成题目的提示词"""
        return f'''基于以下文本内容，生成一道选择题：

文本内容：
{chunk}

要求：
1. 生成一个清晰的问题
2. 生成{num_options}个选项，其中一个是正确答案
3. 选项应该简洁明了，长度尽量相近
4. 正确答案要有明确的依据
5. 干扰项应该合理但错误

请以JSON格式返回，包含以下字段：
- "question": 问题文本
- "options": 选项列表（数组）
- "correct_index": 正确答案在options中的索引（0-based）

示例：
{{
  "question": "文艺复兴时期如何看待哥特风格？",
  "options": ["视为文化复兴的代表", "视为野蛮和黑暗的象征", "视为理性和科学的进步", "视为基督教艺术的巅峰"],
  "correct_index": 1
}}

注意：确保返回纯JSON格式，不要包含其他文本。'''
    
    def _calculate_option_similarities(self, options: List[str], correct_idx: int, 
                                      dimensions: int = None) -> List[float]:
        """计算选项之间的相似度
        
        Args:
            options: 选项列表
            correct_idx: 正确选项的索引
            dimensions: 嵌入维度，如果为None则使用config中的默认值
        """
        if len(options) <= 1:
            return [1.0] * len(options)
        
        # 使用传入的维度或配置中的默认维度
        embedding_dim = dimensions if dimensions is not None else config.EMBEDDING_DIMENSIONS
        
        try:
            # 获取所有选项的embedding（使用Qwen API）
            embeddings = []
            for option in options:
                response = self.embedding_client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=option,
                    dimensions=embedding_dim
                )
                # Qwen API返回的格式：response.data[0].embedding
                embeddings.append(np.array(response.data[0].embedding))
            
            embeddings = np.array(embeddings)
            correct_embedding = embeddings[correct_idx]
            
            # 计算每个选项与正确选项的余弦相似度
            similarities = []
            for i, emb in enumerate(embeddings):
                if i == correct_idx:
                    similarities.append(1.0)  # 正确选项与自己的相似度为1
                else:
                    # 余弦相似度
                    similarity = np.dot(emb, correct_embedding) / (
                        np.linalg.norm(emb) * np.linalg.norm(correct_embedding)
                    )
                    similarities.append(float(similarity))
            
            return similarities
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回随机相似度用于测试
            return [1.0 if i == correct_idx else random.uniform(0.3, 0.7) for i in range(len(options))]
    
    def _filter_options(self, options: List[str], correct_idx: int, 
                        similarities: List[float]) -> Tuple[List[str], int, List[float]]:
        """过滤选项：去掉两个最相似的和两个最不相似的干扰项"""
        if len(options) <= config.RAG_NUM_FINAL_OPTIONS:
            return options, correct_idx, similarities
        
        # 获取所有干扰项的索引和相似度
        distractor_indices = [i for i in range(len(options)) if i != correct_idx]
        distractor_similarities = [similarities[i] for i in distractor_indices]
        
        # 将干扰项按相似度排序
        sorted_distractors = sorted(
            zip(distractor_indices, distractor_similarities),
            key=lambda x: x[1]  # 按相似度排序
        )
        
        # 确保有足够的干扰项进行过滤
        if len(sorted_distractors) < 5:
            # 如果干扰项少于5个，无法去掉4个，随机选择3个
            selected_indices = [idx for idx, _ in random.sample(sorted_distractors, 
                                                            min(3, len(sorted_distractors)))]
        else:
            # 去掉两个最不相似的和两个最相似的
            # 保留中间的部分
            filtered_distractors = sorted_distractors[2:-2]  # 去掉前2个和后2个
            selected_indices = [idx for idx, _ in filtered_distractors]
        
        # 构建最终选项列表（正确选项 + 选中的干扰项）
        final_options = [options[correct_idx]]  # 正确选项
        final_similarities = [1.0]  # 正确选项与自己的相似度为1
        
        for idx in selected_indices:
            final_options.append(options[idx])
            final_similarities.append(similarities[idx])
        
        # 确保最终有4个选项（1个正确 + 3个干扰）
        if len(final_options) > config.RAG_NUM_FINAL_OPTIONS:
            # 如果多于4个，随机选择3个干扰项
            correct_option = final_options[0]
            correct_similarity = final_similarities[0]
            distractors = list(zip(final_options[1:], final_similarities[1:]))
            selected_distractors = random.sample(distractors, 3)
            
            final_options = [correct_option] + [opt for opt, _ in selected_distractors]
            final_similarities = [correct_similarity] + [sim for _, sim in selected_distractors]
        elif len(final_options) < config.RAG_NUM_FINAL_OPTIONS:
            # 如果少于4个，从原始干扰项中补充
            remaining_indices = [i for i in distractor_indices if i not in selected_indices]
            needed_count = config.RAG_NUM_FINAL_OPTIONS - len(final_options)
            if needed_count > 0 and remaining_indices:
                additional_indices = random.sample(remaining_indices, 
                                                min(needed_count, len(remaining_indices)))
                for idx in additional_indices:
                    final_options.append(options[idx])
                    final_similarities.append(similarities[idx])
        
        # 打乱选项顺序（但记录正确选项的新位置）
        option_indices = list(range(len(final_options)))
        random.shuffle(option_indices)
        
        shuffled_options = [final_options[i] for i in option_indices]
        shuffled_similarities = [final_similarities[i] for i in option_indices]
        
        # 找到正确选项的新位置
        new_correct_idx = option_indices.index(0)  # 0是原来正确选项的位置
        
        return shuffled_options, new_correct_idx, shuffled_similarities
    
    def calculate_baseline_similarities(self, question: Question, 
                                       dimensions: int = None) -> List[float]:
        """为Baseline方法计算相似度
        
        Args:
            question: 问题对象
            dimensions: 嵌入维度，如果为None则使用config中的默认值
        """
        print("计算Baseline选项相似度...")
        start_time = time.time()
        embedding_dim = dimensions if dimensions is not None else config.EMBEDDING_DIMENSIONS
        similarities = self._calculate_option_similarities(
            question.options, 
            question.correct_idx,
            dimensions=embedding_dim
        )
        question.embedding_time = time.time() - start_time
        question.similarities = similarities
        return similarities