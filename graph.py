import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

def calculate_cfu_and_plot_from_excel(excel_file='test.xlsx', sheet_name=0, dilution_power_col=None, colony_start_col=1, output_png='cfu_bar_log.png'):
    """
    test.xlsx 파일 처리: 희석배수(10의 n제곱), colony → CFU → log10 바 그래프.
    - 예시 구조: 열0=그룹(Br), 열1~ = colony (희석 dil1=10^1 등 자동).
    - dilution_power_col: 희석 제곱수 열 지정 (None 시 자동 1,2,3...).
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # 그룹 (첫 열)
    groups = df.iloc[:, 0].dropna().tolist()
    
    # Colony 데이터 (colony_start_col부터)
    colony_data = df.iloc[:, colony_start_col:].dropna(how='all').values.astype(float)
    
    # 희석 제곱수 (자동 또는 지정 열)
    if dilution_power_col is None:
        n_dils = colony_data.shape[1]
        dilution_powers = list(range(1, n_dils + 1))  # [1,2,3,4] → 10^1,10^2...
    else:
        dilution_powers = df.iloc[:, dilution_power_col].dropna().tolist()
    
    dilutions = [10 ** p for p in dilution_powers]
    
    # 각 그룹 CFU 계산 (최대값 기준, 평균/중앙값으로 변경 가능)
    cfu_values = []
    for i, group in enumerate(groups):
        if i < colony_data.shape[0]:
            cfus = colony_data[i] * np.array(dilutions)
            max_cfu = np.nanmax(cfus)
            cfu_values.append(max_cfu)
    
    log_cfu = np.log10(cfu_values)
    
    # 바 그래프
    fig = go.Figure(data=go.Bar(x=groups, y=log_cfu, marker_color='steelblue'))
    fig.update_layout(
        title='Endolysin CFU: log10 스케일 (test.xlsx)',
        xaxis_title='실험군',
        yaxis_title='log10(CFU)',
        yaxis_type='log'
    )
    fig.write_image(output_png)
    fig.show()
    
    # 결과 저장
    result_df = pd.DataFrame({'Group': groups, 'Max_CFU': cfu_values, 'log10_CFU': log_cfu})
    result_df.to_excel('cfu_results.xlsx', index=False)
    print(f"그래프: {output_png}, 결과: cfu_results.xlsx")

# 사용: test.xlsx 업로드 후 실행
calculate_cfu_and_plot_from_excel('test.xlsx')
