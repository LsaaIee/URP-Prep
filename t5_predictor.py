import pandas as pd
import numpy as np
import plotly.graph_objects as go  # pip install plotly kaleido

def cfu_plot_csv(csv_file='test.CSV', output='cfu_log.png'):
    df = pd.read_csv(csv_file)
    groups = df.iloc[:, 0].dropna().tolist()  # 첫 열: 그룹 (Br 등)
    colony_data = df.iloc[:, 1:].values.astype(float)  # 나머지: colony
    
    n_dils = colony_data.shape[1]
    dilutions = [10**i for i in range(1, n_dils+1)]  # 10^1, 10^2...
    
    cfu_max = [np.nanmax(colony_data[i] * dilutions) for i in range(len(groups))]
    log_cfu = np.log10(cfu_max)
    
    fig = go.Figure(go.Bar(x=groups, y=log_cfu))
    fig.update_layout(title='log10 CFU', yaxis_type='log', yaxis_title='log10(CFU)')
    fig.write_image(output)
    fig.show()
    
    pd.DataFrame({'Group':groups, 'log10_CFU':log_cfu}).to_csv('results.csv')
    print("완료:", output)

cfu_plot_csv('test.CSV')
