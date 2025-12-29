#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试matplotlib中文字体配置"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 字体文件路径
font_path = '/home/dataset-assist-0/rzh/projects/EOQC_pq/venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'

print("=" * 60)
print("中文字体测试")
print("=" * 60)

# 检查字体文件是否存在
if os.path.exists(font_path):
    print(f"✓ 找到字体文件: {font_path}")
    
    # 方法1: 直接使用字体文件路径（推荐方法）
    try:
        # 创建字体属性对象，直接指定字体文件
        chinese_font = fm.FontProperties(fname=font_path)
        font_name = chinese_font.get_name()
        print(f"✓ 字体名称: {font_name}")
        
        # 创建测试图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 测试中文显示 - 直接使用FontProperties对象
        ax.text(0.5, 0.7, '中文字体测试', ha='center', va='center', 
                fontsize=24, fontweight='bold', fontproperties=chinese_font)
        ax.text(0.5, 0.5, '你好世界！这是SimHei字体', ha='center', va='center', 
                fontsize=18, fontproperties=chinese_font)
        ax.text(0.5, 0.3, '测试内容：哥特建筑、文艺复兴、相似度分析', 
                ha='center', va='center', fontsize=16, fontproperties=chinese_font)
        
        ax.set_title('Matplotlib中文字体测试', fontsize=20, pad=20, fontproperties=chinese_font)
        ax.axis('off')
        
        # 保存图片
        output_file = 'font_test.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 测试图表已保存为: {output_file}")
        print("  请检查图片中的中文是否正常显示（不是方块）")
        
        # 方法2: 尝试将字体添加到字体管理器（用于全局设置）
        print("\n尝试将字体注册到matplotlib...")
        try:
            # 将字体文件添加到字体管理器
            fm.fontManager.addfont(font_path)
            # 重新构建字体缓存
            # 注意：这可能需要一些时间
            print("✓ 字体已添加到字体管理器")
            
            # 现在可以尝试通过名称使用
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 再次测试
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.text(0.5, 0.5, '使用rcParams全局设置测试', ha='center', va='center', 
                    fontsize=20)
            ax2.set_title('全局字体设置测试', fontsize=18)
            ax2.axis('off')
            
            output_file2 = 'font_test_global.png'
            plt.savefig(output_file2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ 全局设置测试图表已保存为: {output_file2}")
            
        except Exception as e2:
            print(f"⚠ 注册字体到管理器失败（不影响直接使用）: {str(e2)}")
            print("  建议：直接使用FontProperties(fname=...)方法")
        
    except Exception as e:
        print(f"✗ 加载字体失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"✗ 字体文件不存在: {font_path}")
    print("请检查路径是否正确")

# 方法2: 尝试通过字体名称查找
print("\n" + "-" * 60)
print("尝试通过字体名称查找...")
try:
    # 清除matplotlib字体缓存（如果需要）
    # fm._rebuild()
    
    # 查找SimHei字体
    fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'simhei' in f.name.lower()]
    if fonts:
        print(f"✓ 在字体列表中找到: {fonts}")
    else:
        print("⚠ 在字体列表中未找到SimHei，可能需要重建字体缓存")
        print("  可以尝试运行: python -c 'import matplotlib.font_manager; matplotlib.font_manager._rebuild()'")
except Exception as e:
    print(f"✗ 查找字体失败: {str(e)}")

print("=" * 60)

