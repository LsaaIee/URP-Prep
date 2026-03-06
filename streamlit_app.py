import streamlit as st
import pandas as pd
from t5_predictor import calculate_final_score, calculate_peptide_properties

st.set_page_config(page_title="T5 Activity Predictor", page_icon="🧬", layout="wide")

st.title("🧬 T5 Endolysin Activity Predictor")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    use_alphafold = st.checkbox("AlphaFold2 자동 실행", value=False, 
                                help="체크하면 구조 예측 자동 실행 (시간 소요)")
    
    st.markdown("### 가중치 조정")
    w_structure = st.slider("구조 안정성", 0.0, 1.0, 0.30, 0.05)
    w_penetration = st.slider("외막 투과성", 0.0, 1.0, 0.40, 0.05)
    w_sequence = st.slider("서열 적합성", 0.0, 1.0, 0.30, 0.05)
    
    total_weight = w_structure + w_penetration + w_sequence
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ 가중치 합이 {total_weight:.2f}입니다. 1.0이 되도록 조정하세요.")

# 메인 영역
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 서열 입력")
    
    full_sequence = st.text_area(
        "전체 T5 서열 (fusion peptide 포함)",
        height=150,
        placeholder="MKTL...",
        help="FASTA 형식 또는 순수 아미노산 서열"
    )
    
    # FASTA 형식 처리
    if full_sequence.startswith(">"):
        lines = full_sequence.split("\n")
        full_sequence = "".join([line for line in lines if not line.startswith(">")])
    
    full_sequence = full_sequence.replace(" ", "").replace("\n", "").upper()

with col2:
    st.header("🔧 Fusion Peptide")
    
    num_fusions = st.number_input(
        "Fusion peptide 개수", 
        min_value=0, max_value=3, value=1, step=1,
        help="0이면 native T5만, 1 이상이면 여러 fusion 조합"
    )

    fusion_peptides = []

    for i in range(num_fusions):
        st.markdown(f"#### Fusion #{i+1}")

        f_type = st.selectbox(
            f"종류 #{i+1}",
            ["KWK", "SMAP-29", "Cys", "Custom"],
            key=f"type_{i}"
        )

        f_seq = st.text_input(
            f"Fusion peptide 서열 #{i+1}",
            key=f"seq_{i}",
            help="해당 fusion peptide 서열만 입력"
        )

        f_pos = st.selectbox(
            f"위치 #{i+1}",
            ["C-terminal", "N-terminal", "Internal"],
            key=f"pos_{i}"
        )

        if f_seq:
            fusion_peptides.append({
                "name": f_type,
                "seq": f_seq,
                "position": f_pos,
            })

    if not use_alphafold:
        manual_plddt = st.number_input(
            "평균 pLDDT (AlphaFold 결과)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.1,
            help="AlphaFold2로 얻은 pLDDT 평균값"
        )
    else:
        manual_plddt = None

    # fusion_type = st.selectbox(
    #     "종류",
    #     ["KWK", "SMAP-29", "Cys", "Custom"]
    # )
    
    # if fusion_type == "Custom":
    #     fusion_peptide = st.text_input("Fusion peptide 서열")
    #     fusion_type_name = st.text_input("이름", "Custom")
    # else:
    #     fusion_peptide = st.text_input(
    #         "Fusion peptide 서열",
    #         help="전체 서열에서 fusion peptide 부분만 입력"
    #     )
    #     fusion_type_name = fusion_type
    
    # fusion_position = st.selectbox(
    #     "위치",
    #     ["C-terminal", "N-terminal", "Internal"]
    # )
    
    # if not use_alphafold:
    #     manual_plddt = st.number_input(
    #         "평균 pLDDT (AlphaFold 결과)",
    #         min_value=0.0,
    #         max_value=100.0,
    #         value=75.0,
    #         step=0.1,
    #         help="AlphaFold2로 얻은 pLDDT 평균값"
    #     )
    # else:
    #     manual_plddt = None

st.markdown("---")

# 예측 버튼
if st.button("🚀 예측 시작", type="primary", use_container_width=True):
    
    if not full_sequence: #or not fusion_peptide:
        st.error("❌ 서열을 입력해주세요!")
    else:
        with st.spinner("분석 중... (최대 2-3분 소요)"):
            try:
                weights = {
                    "structure": w_structure,
                    "penetration": w_penetration,
                    "sequence": w_sequence,
                }

                result = calculate_final_score(
                    sequence=full_sequence,
                    fusion_peptides=fusion_peptides, # singular to plural
                    # fusion_type=fusion_type_name,
                    manual_plddt=manual_plddt,
                    use_alphafold=use_alphafold,
                    weights=weights,
                )
                
                # 결과 표시
                st.success("✅ 분석 완료!")
                
                # 스코어 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "최종 스코어",
                        f"{result['final_score']:.3f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "구조 안정성",
                        f"{result['structure_score']:.3f}",
                        delta=f"pLDDT: {result['plddt']:.1f}"
                    )
                
                with col3:
                    st.metric(
                        "외막 투과성",
                        f"{result['penetration_score']:.3f}"
                    )
                
                with col4:
                    st.metric(
                        "서열 적합성",
                        f"{result['sequence_score']:.3f}",
                        delta=f"MLM: {result['mlm_loss']:.2f}"
                    )
                
                st.markdown("---")
                
                # 의사결정
                decision = result['decision']
                
                if "PROCEED" in decision:
                    st.success(f"### {decision}")
                    st.balloons()
                elif "TEST 2-3" in decision:
                    st.warning(f"### {decision}")
                elif "CAUTION" in decision:
                    st.info(f"### {decision}")
                else:
                    st.error(f"### {decision}")
                
                st.write(f"**설명:** {result['explanation']}")
                st.write(f"**예상 효과:** {result['expected_cfu']}")
                
                # 상세 정보
                with st.expander("📊 상세 분석 결과"):                    
                    # Fusion peptide 특성
                    # props = calculate_peptide_properties(fusion_peptide)
                    # if props:
                    #     st.subheader("Fusion Peptide 특성")
                        
                    #     prop_col1, prop_col2, prop_col3 = st.columns(3)
                        
                    #     with prop_col1:
                    #         st.metric("순전하 (pH 7.4)", f"{props['charge']:+.2f}")
                    #         st.metric("분자량", f"{props['molecular_weight']:.1f} Da")
                        
                    #     with prop_col2:
                    #         st.metric("소수성 (GRAVY)", f"{props['gravy']:.3f}")
                    #         st.metric("양전하 비율", f"{props['positive_ratio']:.1%}")
                        
                    #     with prop_col3:
                    #         st.metric("소수성 AA 비율", f"{props['hydrophobic_ratio']:.1%}")
                    
                    st.subheader("Fusion Peptide 특성")

                    if not fusion_peptides:
                        st.write("Fusion peptide 없음 (T5).")
                    else: 
                        for i, fp in enumerate(fusion_peptides):
                            st.markdown(f"#### Fusion #{i+1} - {fp['name']} ({fp['position']})")
                            props = calculate_peptide_properties(fp["seq"])
                            if props:
                                prop_col1, prop_col2, prop_col3 = st.columns(3)

                                with prop_col1:
                                    st.metric("순전하 (pH 7.4)", f"{props['charge']:+.2f}")
                                    st.metric("분자량", f"{props['molecular_weight']:.1f} Da")
                                with prop_col2:
                                    st.metric("소수성 (GRAVY)", f"{props['gravy']:.3f}")
                                    st.metric("양전하 비율", f"{props['positive_ratio']:.1%}")
                                with prop_col3:
                                    st.metric("소수성 AA 비율", f"{props['hydrophobic_ratio']:.1%}")

                    # 가중치 정보
                    st.subheader("가중치 분배")
                    weights = result['weights']
                    st.write(f"- 구조: {weights['structure']:.0%}")
                    st.write(f"- 투과: {weights['penetration']:.0%}")
                    st.write(f"- 서열: {weights['sequence']:.0%}")
                
                # 결과 다운로드
                result_df = pd.DataFrame([result])
                csv = result_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 결과 다운로드 (CSV)",
                    data=csv,
                    file_name="prediction_result.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
                st.exception(e)

# 배치 예측
st.markdown("---")
st.header("📦 배치 예측")

uploaded_file = st.file_uploader(
    "CSV 파일 업로드",
    type=['csv'],
    help="형식: name,sequence,fusion_peptide,fusion_type,plddt"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("미리보기:", df.head())
    
    if st.button("배치 예측 시작"):
        st.info("구현 예정...")

# 사용 가이드
with st.expander("📖 사용 가이드"):
    st.markdown("""
    ### 입력 형식
    
    1. **전체 T5 서열**: Fusion peptide가 포함된 완전한 서열
    2. **Fusion peptide**: 부착된 펩타이드 서열만
    3. **pLDDT**: AlphaFold2로 얻은 평균 pLDDT 값 (50-100)
    
    ### 스코어 해석
    
    - **≥0.75**: 즉시 LNP 단계 진행
    - **0.60-0.74**: CFU counting 2-3회 후 결정
    - **0.45-0.59**: 조건 최적화 고려
    - **<0.45**: 서열 재설계 권장
    
    ### 예시
    
    ```
    T5 서열: MKTL...RGLRRLGRKIAHGVKKY
    Fusion peptide: RGLRRLGRKIAHGVKKY (SMAP-29)
    pLDDT: 82.5
    ```
    """)
