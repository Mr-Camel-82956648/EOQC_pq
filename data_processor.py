# data_processor.py
import re
from typing import List, Tuple

class DataProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def get_sample_text(self) -> str:
        """返回固定的示例文本"""
        return """哥特建筑不适用于那些根据古典和文艺复兴建筑 建立起来的范畴与概念 -- 它奇特而与众不同，易于 复制却难以理解。虽然荒谬，但是哥特这个称呼却被 保留下来；没有其他任何建筑时期有这么不合时宜的 称呼。这些是怎样发生的？即使是对于最出色和努力 研究者来说，哥特建筑也难以捉摸。它的神秘，它的 原始能量似乎体现在 "哥特" 这个称呼中，因为这个 名称中蕴含着北方野蛮部落那种神秘起源、传说中的 /n漫游、野性的想象等意味。最终 "哥特" 这个称呼不 是直接定义建筑特性，与此相反，这个名字的内涵被 这种建筑或者是其他任何能从中读出或得出的意义所 定义 (于是有 "哥特" 小说，晦涩而怪异)。/n现代建筑历史学家已经走到了过去简单化解释的 对立面，创造出各种令人困惑的解释。总体上有三种 解释路径占统治地位：结构、视觉以及象征性。第一 种强调哥特建筑的石质结构骨架，哥特建筑被看做完 全由工程逻辑所控制，就仿佛它的建造者没有其他任 何想法，一心只想把厚重的罗马风墙壁减小为最细的 石质骨架。这一观点的 19 世纪支持者将哥特建筑与他 们以铸铁和钢为基础的新建筑技术联系起来。"""
    
    def split_into_chunks(self, text: str, num_chunks: int) -> List[str]:
        """将文本分割成指定数量的chunk（简化版本）"""
        # 简单的段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # 合并小段落
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 如果chunks数量不够，就返回整个文本作为一个chunk
        if len(chunks) < num_chunks:
            return [text] * num_chunks
        
        return chunks[:num_chunks]