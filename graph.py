import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import random
import numpy as np

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

file_path = 'graph_test.xlsx' 
df = pd.read_excel(file_path)

x_data = df['실험군']
y_data = df['평균']
num_bars = len(x_data)

def format_scientific(value, tick_number):
    if value == 0:
        return "0"
    exponent = int(math.log10(abs(value))) # 10의 제곱수 계산
    
    return f"$10^{exponent}$"

grays = np.linspace(0.3, 0.8, num_bars)
grayscale_colors = [str(val) for val in grays]
random.shuffle(grayscale_colors)

plt.figure(figsize=(10, 6))

bars = plt.bar(x_data, y_data, color=grayscale_colors, edgecolor='black', alpha=0.9, width=0.5)

plt.title('실험군별 CFU 평균', fontsize=16, pad=15)
plt.xlabel('실험군', fontsize=12)
plt.ylabel('CFU/ml (평균)', fontsize=12)

plt.xticks(rotation=45, ha='right') 
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_scientific))

plt.tight_layout()
plt.savefig('cfu_graph_output.png', dpi=300, bbox_inches='tight')
plt.show()