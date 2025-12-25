# config.py
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class Config:
    # API配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 千问API密钥
    GITEEAI_API_KEY = os.getenv("GITEEAI_API_KEY")  # 模力方舟API密钥
    
    # 实验参数
    NUM_QUESTIONS = 1  # 单次实验，先测试1道题
    BASELINE_NUM_OPTIONS = 4
    RAG_NUM_CANDIDATES = 8
    RAG_NUM_FINAL_OPTIONS = 4
    
    # 模型配置
    EMBEDDING_MODEL = "text-embedding-v4"  # Qwen嵌入模型
    EMBEDDING_DIMENSIONS = 64  # 嵌入维度，可调整用于比较不同维度对相似度的影响
    GENERATION_MODEL = "qwen3-max"  # 使用qwen3-max
    GENERATION_API = "qwen"  # 指定为qwen
    
    # 文本分割配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # 实验设置
    RANDOM_SEED = 42
    TEMPERATURE = 0.7

config = Config()