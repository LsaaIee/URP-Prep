import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def calculate_cfu_and_plot(csv_file, dilution_cols=None, output_png='cfu_bar_log.png'):
    """
    CSV 파일을 읽어 CFU 계산 후 log10 바 그래프 생성.
    - CSV 형식 예시:
      그룹,dil1,dil2,dil3,dil4
      Br,13,15,17,1
      TS5KW0.25uM,15,13,17,3
      ...
    - dil1=10^1 (10배 희석), dil2=10^2 등으로 자동 계산 (CFU = colony * 10^dil).
    """
    df = pd.read_csv(csv_file)
    
    # 그룹 이름 (첫 행)
    groups = df.iloc[0, 1:].tolist()  # 첫 행의 나머지 열: 실험군 (TS5KW 등)
    df_groups = df.iloc[1:, 0].tolist()  # 첫 열의 나머지: 그룹 (Br 등)
    
    # 데이터 추출 (희석별 colony)
    data_matrix = df.iloc[1:, 1:].values.astype(float)
    
    # 희석배수 자동 설정 (dil1=10^1, dil2=10^2 등; 사용자 지정 가능)
    if dilution_cols is None:
        n_dils = data_matrix.shape[1]
        dilutions = [10**i for i in range(1, n_dils + 1)]
    else:
        dilutions = dilution_cols  # 예: [10, 100, 1000]
    
    # 각 그룹별 max CFU (각 희석에서 CFU 계산 후 최대값 사용, 일반적 방법)
    cfu_values = []
    for row_idx, group_name in enumerate(df_groups):
        cfus = data_matrix[row_idx] * np.array(dilutions)
        max_cfu = np.max(cfus)  # 또는 np.mean(cfus) 등으로 변경 가능
        cfu_values.append(max_cfu)
    
    # log10 변환 (y축)
    log_cfu = np.log10(cfu_values)
    
    # Plotly 바 그래프
    fig = go.Figure(data=[
        go.Bar(x=groups, y=log_cfu, marker_color='steelblue', name='log10(CFU)')
    ])
    fig.update_layout(
        title='Endolysin 활성도: log10 CFU (최대값 기준)',
        xaxis_title='실험군 (Concentration)',
        yaxis_title='log10(CFU)',
        yaxis_type='log',  # 10의 제곱수 스케일
        template='plotly_white'
    )
    
    # PNG 저장 및 show
    fig.write_image(output_png)
    fig.show()
    
    # 결과 CSV 저장 (계산된 CFU)
    result_df = pd.DataFrame({'Group': df_groups, 'Max_log10_CFU': log_cfu, 'Max_CFU': cfu_values})
    result_df.to_csv('cfu_results.csv', index=False)
    print("그래프 저장:", output_png)
    print("결과 CSV:", 'cfu_results.csv')
    print(result_df)

# 사용 예시 (사용자 CSV 파일 경로 입력)
# calculate_cfu_and_plot('your_data.csv', dilutions=[10, 100, 1000, 10000])
