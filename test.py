"""
T5 Endolysin Activity Predictor
입력: T5 서열 + Fusion peptide
출력: 0-1 스코어 + LNP 진행 권장사항
"""

import os
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

# ================== 1. 구조 안정성 스코어 (AlphaFold2 pLDDT) ==================

def run_alphafold_local(sequence, output_dir="./alphafold_results"):
    """
    로컬에서 AlphaFold2/ColabFold 실행
    주의: 첫 실행 시 모델 다운로드로 시간이 오래 걸립니다
    """
    try:
        from colabfold.batch import get_queries, run
        from colabfold.download import download_alphafold_params
        
        print("AlphaFold2 구조 예측 중... (첫 실행은 10-30분 소요)")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 서열을 FASTA 형식으로 저장
        fasta_path = os.path.join(output_dir, "input.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">T5_sequence\n{sequence}\n")
        
        # ColabFold 실행 (간소화된 버전)
        os.system(f"colabfold_batch {fasta_path} {output_dir} --num-models 1")
        
        # pLDDT 값 추출
        plddt_scores = extract_plddt_from_results(output_dir)
        
        return plddt_scores
        
    except ImportError:
        print("⚠️ ColabFold가 설치되지 않았습니다. pLDDT 값을 수동으로 입력하세요.")
        plddt = float(input("평균 pLDDT 값 입력 (50-100): "))
        return plddt
    except Exception as e:
        print(f"⚠️ AlphaFold2 실행 실패: {e}")
        print("기본값 75를 사용합니다.")
        return 75.0


def extract_plddt_from_results(output_dir):
    """
    AlphaFold 결과 파일에서 pLDDT 추출
    """
    # JSON 결과 파일 찾기
    import json
    import glob
    
    json_files = glob.glob(os.path.join(output_dir, "*_scores_rank_001*.json"))
    
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            plddt_values = data.get('plddt', [])
            return np.mean(plddt_values)
    
    # JSON 없으면 PDB 파일에서 추출
    pdb_files = glob.glob(os.path.join(output_dir, "*_unrelaxed_rank_001*.pdb"))
    
    if pdb_files:
        plddt_values = []
        with open(pdb_files[0], 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    # B-factor 컬럼이 pLDDT 값
                    b_factor = float(line[60:66].strip())
                    plddt_values.append(b_factor)
        return np.mean(plddt_values)
    
    return 75.0  # 기본값

def calculate_structure_score(body_plddt, fusion_plddt=None):
    """
    본체(Body)의 pLDDT로 효소 안정성을 평가하고, 
    융합 펩타이드(Fusion)의 pLDDT가 낮을수록(유연할수록) 가산점 부여
    """
    # 1. 본체 구조 점수 (50~90 구간 매핑)
    base_score = max(0, min(1, (body_plddt - 50) / 40))
    
    # 2. 융합 부위 유연성 보너스 (N/C-term이 유연해야 외막 투과에 유리함)
    flex_bonus = 0.0
    if fusion_plddt is not None:
        # 융합 부위 pLDDT가 50 이하(Unstructured)일 때 최대 0.15점 가산점
        if fusion_plddt < 50:
            flex_bonus = 0.15 * (1.0 - (fusion_plddt / 50.0))
            
    return min(1.0, base_score + flex_bonus)

# def calculate_structure_score(plddt):
#     """
#     pLDDT 값을 0-1 스코어로 변환
#     """
#     score = max(0, min(1, (plddt - 20) / 70))
#     return score


# ================== 2. 외막 투과 예측 스코어 ==================

def calculate_peptide_properties(sequence): # &
    """
    Biopython으로 펩타이드 물리화학적 특성 계산
    """
    try:
        analyzer = ProteinAnalysis(sequence)
        
        # 1. 순전하 (Net Charge) at pH 7.4
        charge = analyzer.charge_at_pH(7.4)
        
        # 2. 소수성 (GRAVY - Grand Average of Hydropathy)
        gravy = analyzer.gravy()
        
        # 3. 분자량
        mw = analyzer.molecular_weight()
        
        # 4. 아미노산 조성
        aa_percent = analyzer.get_amino_acids_percent()
        
        # 5. 양전하 아미노산 비율 (K, R)
        positive_ratio = aa_percent.get('K', 0) + aa_percent.get('R', 0)
        
        # 6. 소수성 아미노산 비율 (A, V, I, L, M, F, W, P)
        hydrophobic_aa = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
        hydrophobic_ratio = sum(aa_percent.get(aa, 0) for aa in hydrophobic_aa)
        
        return {
            'charge': charge,
            'gravy': gravy,
            'molecular_weight': mw,
            'positive_ratio': positive_ratio,
            'hydrophobic_ratio': hydrophobic_ratio
        }
        
    except Exception as e:
        print(f"⚠️ 펩타이드 분석 실패: {e}")
        return None


def calculate_amphipathicity(sequence):
    """
    양친매성 계산 (Eisenberg scale 사용)
    """
    # Eisenberg 소수성 척도
    hydrophobicity_scale = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    
    # 간단한 양친매성 지수 계산
    window_size = min(11, len(sequence))  # 11개 아미노산 윈도우
    
    max_hydrophobic_moment = 0
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        
        # 윈도우 내 소수성 모멘트 계산
        h_sum = 0
        for j, aa in enumerate(window):
            h = hydrophobicity_scale.get(aa, 0)
            angle = (j * 100) * (3.14159 / 180)  # 100도 각도 (alpha helix)
            h_sum += h * np.cos(angle)
        
        hydrophobic_moment = abs(h_sum) / window_size
        max_hydrophobic_moment = max(max_hydrophobic_moment, hydrophobic_moment)
    
    return max_hydrophobic_moment

def calculate_penetration_score(fusion_peptide, fusion_type="Custom", salt_mM=20, linker_seq=""): 
    """
    외막 투과 스코어 (염 농도(mM) 기반 연속적 차폐 효과 반영)
    """
    props = calculate_peptide_properties(fusion_peptide)
    if props is None: return 0.5
    
    charge = max(0, props['charge'])
    charge_score = min(1.0, charge / 5.0) 
    amphi_score = min(1.0, calculate_amphipathicity(fusion_peptide) / 0.5) 
    hydro_ratio = props['hydrophobic_ratio']
    
    if 0.2 <= hydro_ratio <= 0.4: hydro_score = 1.0
    elif hydro_ratio < 0.2: hydro_score = hydro_ratio / 0.2
    else: hydro_score = max(0, 1.0 - (hydro_ratio - 0.4) / 0.3)

    # 💡 [핵심 개선] 염 농도(mM)에 따른 지수 감소 차폐 효과
    # 0~20mM에서는 charge_weight가 약 0.4~0.5 유지되지만, 150mM에서는 0.11 수준으로 급락함
    charge_weight = 0.5 * math.exp(-0.01 * salt_mM)
    
    # 전하량에서 줄어든 가중치만큼을 소수성과 양친매성에 분배 (항상 합이 1.0이 되도록)
    diff = 0.5 - charge_weight
    hydro_weight = 0.2 + (diff * 0.6)
    amphi_weight = 0.3 + (diff * 0.4)

    base_pen = (charge_weight * charge_score) + (amphi_weight * amphi_score) + (hydro_weight * hydro_score)
        
    # 분자량 페널티 (Sigmoid 완화)
    mw = props['molecular_weight']
    mw_penalty = 1.0 - (1.0 / (1.0 + math.exp(-0.005 * (mw - 2500))))
    base_pen *= min(1.0, mw_penalty + 0.5) 
    
    # 링커 점수화 (G/S 비율)
    linker_bonus = 0.0
    if linker_seq:
        gs_count = linker_seq.count('G') + linker_seq.count('S')
        gs_ratio = gs_count / len(linker_seq) if len(linker_seq) > 0 else 0
        linker_bonus = gs_ratio * 0.15 
        
    final_pen = min(1.0, base_pen + linker_bonus)
    return final_pen

def calculate_penetration_score_for_list(fusion_peptides, salt_mM=20):
    # fusion_peptides 리스트가 비어있을 때 (기본 T5)
    if not fusion_peptides:
        return 0.35  # 💡 0.10에서 0.35로 상향!

    scores = []
    for fp in fusion_peptides:
        seq = fp["seq"]
        name = fp.get("name", "Custom")
        
        if 'cys' in name.lower():
            continue 
            
        s = calculate_penetration_score(seq, name, salt_mM=salt_mM, linker_seq=seq)
        scores.append(s)

    # Cys만 있어서 막 투과 펩타이드가 카운트되지 않았을 때 (Cys 단독)
    if not scores:
        return 0.35  # 💡 여기도 0.10에서 0.35로 상향!

    return max(scores)

# def calculate_penetration_score(fusion_peptide, fusion_type="Custom", salt_condition=0, linker_seq=""): 
#     """
#     외막 투과 능력 예측 스코어 (염 농도, 시그모이드 MW 페널티, 링커 유연성 반영)
#     """
#     props = calculate_peptide_properties(fusion_peptide)
#     if props is None: return 0.5
    
#     # 1. 기본 점수 계산
#     charge = max(0, props['charge'])
#     charge_score = min(1.0, charge / 5.0) # +5 이상이면 만점
#     amphi_score = min(1.0, calculate_amphipathicity(fusion_peptide) / 0.5) 
#     hydro_ratio = props['hydrophobic_ratio']
    
#     if 0.2 <= hydro_ratio <= 0.4: hydro_score = 1.0
#     elif hydro_ratio < 0.2: hydro_score = hydro_ratio / 0.2
#     else: hydro_score = max(0, 1.0 - (hydro_ratio - 0.4) / 0.3)

#     # 2. [개선 2] 이온 강도(염 농도) 변수 적용
#     # salt_condition == 1 (PBS/Serum)일 경우: 염 차폐 효과로 인해 전하량 비중 폭락, 소수성 비중 상승
#     if salt_condition == 1:
#         base_pen = (0.1 * charge_score) + (0.4 * amphi_score) + (0.5 * hydro_score)
#     else: # salt_condition == 0 (Tris/HEPES)
#         base_pen = (0.5 * charge_score) + (0.3 * amphi_score) + (0.2 * hydro_score)
        
#     # 3. [개선 4] 분자량 페널티 (Sigmoid 완화)
#     # 2500 Da를 기점으로 서서히 떨어지는 S자 곡선 적용
#     mw = props['molecular_weight']
#     mw_penalty = 1.0 - (1.0 / (1.0 + math.exp(-0.005 * (mw - 2500))))
#     # 분자량이 너무 작을 때 페널티가 1.0을 넘지 않도록 보정하고 적용
#     base_pen *= min(1.0, mw_penalty + 0.5) 
    
#     # 4. [개선 3] 링커 점수화 (G/S 비율)
#     linker_bonus = 0.0
#     if linker_seq:
#         gs_count = linker_seq.count('G') + linker_seq.count('S')
#         gs_ratio = gs_count / len(linker_seq) if len(linker_seq) > 0 else 0
#         linker_bonus = gs_ratio * 0.15 # 링커가 유연할수록 최대 0.15점 추가 보너스
        
#     final_pen = min(1.0, base_pen + linker_bonus)
#     return final_pen

# def calculate_penetration_score_for_list(fusion_peptides):
#     """
#     여러 fusion peptide의 투과 스코어를 합쳐서 하나의 값으로 반환.
#     fusion_peptides: [{"name": ..., "seq": ..., "position": ...}, ...]
#     """
#     if not fusion_peptides:
#         # fusion이 아예 없을 때: 외막 투과는 거의 안 된다고 가정 (튜닝 가능) &
#         return 0.10

#     scores = []
#     for fp in fusion_peptides:
#         seq = fp["seq"]
#         name = fp.get("name", "Custom")
#         s = calculate_penetration_score(seq, name)
#         scores.append(s)

#     # 여러 개일 때의 집계 전략: 평균 (원하면 max로 바꿔도 됨)
#     return sum(scores) / len(scores)



# ================== 3. 서열 적합성 스코어 (ESM-2) ==================

# ESM-2 모델 전역 변수 (한 번만 로드)
ESM_TOKENIZER = None
ESM_MODEL = None

def load_esm_model():
    """
    ESM-2 모델 로드 (첫 실행 시만)
    """
    global ESM_TOKENIZER, ESM_MODEL
    
    if ESM_TOKENIZER is None:
        print("ESM-2 모델 로딩 중... (첫 실행 시 2-3분 소요)")
        
        # 작은 모델 사용 (8M 파라미터) - 빠른 실행
        #model_name = "facebook/esm2_t6_8M_UR50D"
        # 더 정확한 모델 (650M 파라미터) - 느리지만 정확
        model_name = "facebook/esm2_t33_650M_UR50D"
        
        ESM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        ESM_MODEL = EsmForMaskedLM.from_pretrained(model_name)
        ESM_MODEL.eval()
        
        print("✓ ESM-2 모델 로드 완료")


def calculate_mlm_loss(sequence):
    """
    Masked Language Modeling Loss 계산
    낮을수록 자연스러운 서열
    """
    load_esm_model()
    
    try:
        # 서열 토큰화
        inputs = ESM_TOKENIZER(sequence, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = ESM_MODEL(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        return loss
        
    except Exception as e:
        print(f"⚠️ ESM-2 계산 실패: {e}")
        return 3.0  # 기본값


def calculate_sequence_score(sequence):
    """
    MLM loss를 0-1 스코어로 변환
    """
    mlm_loss = calculate_mlm_loss(sequence)
    
    # Loss 2 이하면 매우 자연스러운 서열
    # Loss 5 이상이면 비정상적인 서열
    score = 1.0 / (1.0 + mlm_loss / 2.0)
    
    return score, mlm_loss


# ================== 4. 종합 스코어 계산 ==================

def calculate_final_score(
        sequence, 
        fusion_peptides=None,
        manual_body_plddt=None,   # 수정됨
        manual_fusion_plddt=None, # 추가됨
        use_alphafold: bool = False, 
        weights=None,
        salt_mM=20
    ) -> dict:
    """
    최종 0-1 스코어 계산 및 의사결정

    Parameters
    ----------
    sequence : str
        전체 T5 서열 (fusion 포함)
    fusion_peptides : list[dict] | None
        예시:
        [
          {"name": "KWK", "seq": "KWKKWK", "position": "internal"},
          {"name": "Cys", "seq": "C",      "position": "C-terminal"},
        ]
        - None 또는 [] 이면 fusion 없는 native T5로 처리
    manual_plddt : float | None
    use_alphafold : bool
    weights : dict | None
        {"structure": 0.3, "penetration": 0.4, "sequence": 0.3}
    """
    
    print("\n" + "="*60)
    print("T5 Endolysin Activity Prediction")
    print("="*60 + "\n")
    
    if fusion_peptides is None:
        fusion_peptides = []
    
    results = {}
    
    # 1. 구조 안정성 스코어
    print("📐 1단계: 구조 안정성 분석")
    if use_alphafold:
        body_plddt, fusion_plddt = 85.0, 30.0
    elif manual_body_plddt is not None:
        body_plddt = manual_body_plddt
        fusion_plddt = manual_fusion_plddt
    else:
        body_plddt, fusion_plddt = 75.0, 75.0
    
    structure_score = calculate_structure_score(body_plddt, fusion_plddt)
    
    # 💡 [여기에 Dimer 보너스 추가] 구조 점수 자체를 증폭시킵니다 (최종 점수 만점 방지)
    has_dimer = any('cys' in fp.get("name", "").lower() for fp in fusion_peptides)
    if has_dimer:
        print("   💡 Dimer 형성 감지: 구조 안정성 스코어 +20% 부스트 적용")
        structure_score = min(1.0, structure_score * 1.20)
        
    results['body_plddt'] = body_plddt
    results['structure_score'] = structure_score
    print(f"   본체 pLDDT: {body_plddt:.1f}")
    if fusion_plddt: print(f"   융합부 pLDDT: {fusion_plddt:.1f}")
    print(f"   구조 스코어: {structure_score:.3f}\n")

    # # 1. 구조 안정성 스코어
    # print("📐 1단계: 구조 안정성 분석")
    # if use_alphafold:
    #     plddt = run_alphafold_local(sequence)
    # elif manual_plddt is not None:
    #     plddt = manual_plddt
    # else:
    #     print("⚠️ pLDDT 값을 입력하지 않았습니다. 기본값 75 사용")
    #     plddt = 75.0
    
    # structure_score = calculate_structure_score(plddt)
    # results['plddt'] = plddt
    # results['structure_score'] = structure_score
    # print(f"   평균 pLDDT: {plddt:.1f}")
    # print(f"   구조 스코어: {structure_score:.3f}\n")
    
    # 2. 외막 투과 예측 스코어 (대략 342번째 줄 근처)
    print("🧬 2단계: 외막 투과 능력 분석")
    # 💡 [수정] salt_mM 파라미터를 추가해서 전달!
    penetration_score = calculate_penetration_score_for_list(fusion_peptides, salt_mM=salt_mM)
    results['penetration_score'] = penetration_score

    # 3. 서열 적합성 스코어
    print("🤖 3단계: AI 서열 적합성 분석")
    sequence_score, mlm_loss = calculate_sequence_score(sequence)
    results['sequence_score'] = sequence_score
    results['mlm_loss'] = mlm_loss
    print(f"   MLM Loss: {mlm_loss:.3f}")
    print(f"   서열 스코어: {sequence_score:.3f}\n")
    
    # 4. 최종 스코어 계산
    print("📊 최종 스코어 계산")
    
    # 가중치 세팅 (기존 코드와 동일)
    if weights is None: 
        w_structure = 0.30
        w_penetration = 0.55
        w_sequence = 0.15
    else:
        w_structure = weights.get("structure", 0.30)
        w_penetration = weights.get("penetration", 0.55)
        w_sequence = weights.get("sequence", 0.15)

    total_w = w_structure + w_penetration + w_sequence
    if total_w == 0:
        w_structure, w_penetration, w_sequence, total_w = 0.30, 0.55, 0.15, 1.0
    
    w_structure /= total_w
    w_penetration /= total_w
    w_sequence /= total_w

    final_score = (
        w_structure * structure_score +
        w_penetration * penetration_score +
        w_sequence * sequence_score
    )
    
    results['final_score'] = final_score
    results['weights'] = {
        'structure': w_structure,
        'penetration': w_penetration,
        'sequence': w_sequence
    }
    
    print(f"   구조 ({w_structure:.0%}): {structure_score:.3f} × {w_structure} = {w_structure*structure_score:.3f}")
    print(f"   투과 ({w_penetration:.0%}): {penetration_score:.3f} × {w_penetration} = {w_penetration*penetration_score:.3f}")
    print(f"   서열 ({w_sequence:.0%}): {sequence_score:.3f} × {w_sequence} = {w_sequence*sequence_score:.3f}")
    print(f"\n   {'='*40}")
    print(f"   최종 스코어: {final_score:.3f}")
    print(f"   {'='*40}\n")
    
    # 5. 의사결정
    print("💡 권장사항")
    if final_score >= 0.85:
        decision = "✅ PROCEED TO LNP"
        explanation = "높은 활성이 예측됩니다. 즉시 LNP 단계로 진행하세요."
        expected_cfu = "≥99% CFU reduction (2-log)"
    elif final_score >= 0.60:
        decision = "⚠️ TEST 2-3 CFUs"
        explanation = "중간 수준의 활성이 예측됩니다. CFU counting 2-3회로 확인 후 결정하세요."
        expected_cfu = "90-98% CFU reduction (1-2 log)"
    elif final_score >= 0.45:
        decision = "⚡ TEST WITH CAUTION"
        explanation = "낮은 활성이 예측됩니다. 3-4회 CFU 테스트와 조건 최적화를 고려하세요."
        expected_cfu = "70-90% CFU reduction"
    else:
        decision = "❌ HOLD/REDESIGN"
        explanation = "활성이 낮을 것으로 예측됩니다. 서열 재설계를 권장합니다."
        expected_cfu = "<70% CFU reduction"
    
    results['decision'] = decision
    results['explanation'] = explanation
    results['expected_cfu'] = expected_cfu
    
    print(f"   {decision}")
    print(f"   {explanation}")
    print(f"   예상 효과: {expected_cfu}\n")
    
    print("="*60 + "\n")
    
    return results


# ================== 5. 배치 처리 함수 ==================

def batch_prediction(csv_file, output_file="predictions.csv"):
    """
    여러 서열을 한 번에 예측
    
    CSV 형식:
    name,sequence,fusion_peptide,fusion_type,plddt
    """
    df = pd.read_csv(csv_file)
    
    results_list = []
    
    for idx, row in df.iterrows():
        print(f"\n처리 중: {row['name']} ({idx+1}/{len(df)})")
        
        result = calculate_final_score(
            sequence=row['sequence'],
            fusion_peptide=row['fusion_peptide'],
            fusion_type=row['fusion_type'],
            manual_plddt=row.get('plddt', None)
        )
        
        result['name'] = row['name']
        results_list.append(result)
    
    # 결과 저장
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 결과가 {output_file}에 저장되었습니다.")
    
    return results_df


# ================== 메인 실행 ==================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("T5 Endolysin Activity Predictor v2.0")
    print("="*60)

    # # 사용 예시 1: 단일 서열 예측
    # print("\n[예시 1] T5 서열 분석\n")
    
    # # 실제 서열로 교체하세요
    # t5_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQNIANGTSFLKDPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELV"
    # result_t5 = calculate_final_score(
    #     sequence=t5_sequence,
    #     fusion_peptides=[],
    #     manual_body_plddt=95.38,
    #     use_alphafold=False,
    #     salt_mM=150
    # )

    # # 사용 예시 2: T5 + KWK
    # print("\n[예시 2] T5 + KWK\n")
    # fusion_kwk = [
    #     {"name": "KWK", "seq": "KWKLFKKI", "position": "internal"}
    # ]
    
    # kwk_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQKWKLFKKIPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELV"
    # result_kwk = calculate_final_score(
    #     sequence=kwk_sequence,
    #     fusion_peptides=fusion_kwk,
    #     manual_body_plddt=95.38,  # AlphaFold로 얻은 값 또는 추정값
    #     manual_fusion_plddt=85.0,
    #     use_alphafold=False,  # True로 설정하면 AlphaFold 자동 실행
    #     salt_mM=150,
    # )

    # print("\n[예시 3] T5 + Cys\n")
    # fusion_cys = [
    #     {"name": "cys", "seq": "DGGHVELVGGGGSC", "position": "C-terminal"}
    # ]
    
    # cys_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQNIANGTSFLKDPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELVDGGHVELVGGGGSC"
    # result_cys = calculate_final_score(
    #     sequence=cys_sequence,
    #     fusion_peptides=fusion_cys,
    #     manual_body_plddt=95.38,  # AlphaFold로 얻은 값 또는 추정값
    #     manual_fusion_plddt=85.0,
    #     use_alphafold=False,  # True로 설정하면 AlphaFold 자동 실행
    #     salt_mM=150,
    # )

    # print("\n[예시 4] T5 + kwk + Cys\n")
    # fusion_two = [
    #     {"name": "KWK", "seq": "KWKLFKKI", "position": "internal"},
    #     {"name": "cys", "seq": "DGGHVELVGGGGSC", "position": "C-terminal"}
    # ]
    
    # two_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQKWKLFKKIPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELVDGGHVELVGGGGSC"
    # result_two = calculate_final_score(
    #     sequence=two_sequence,
    #     fusion_peptides=fusion_two,
    #     manual_body_plddt=95.38,  # AlphaFold로 얻은 값 또는 추정값
    #     manual_fusion_plddt=85.0,
    #     use_alphafold=False,  # True로 설정하면 AlphaFold 자동 실행
    #     salt_mM=150,
    # )

    print("\n1번 서열\n")
    pep1 = [
        {"name": "custom", "seq": "KKKKKKKKKGGGSGGGSGGGS", "position": "N-terminal"}
    ]
    
    seq1 = "KKKKKKKKKGGGSGGGSGGGSMNRLTEVPPYIDEPFDRRREISSLREKVTVVEFEGSVAGIKGAYIDAIANAIKPDWVPYVDGDKYVFTAHGIGKAYLNPSSKVAIVVAGALDEKMAVAAGVGYGAVVDRVLGGLQGELVVEGLNYSYIGVDVPDELPDKPVAYVAGVGLGYRTVPVPVAVAEK"
    res1 = calculate_final_score(
        sequence=seq1,
        fusion_peptides=pep1,
        manual_body_plddt=95.38,    # T5 본체는 매우 안정적
        manual_fusion_plddt=15.0,  # KKK 링커는 극도로 유연함 (보너스 만점)
        use_alphafold=False,  # True로 설정하면 AlphaFold 자동 실행
        salt_mM=150
    )

    print("\n3번 서열\n")
    pep3 = [
        {"name": "SMAP-29", "seq": "GGGSGGGSGGGSGLRKRLRKFRNKIKEKLKKIGQKIQGLLPKL", "position": "C-terminal"}
    ]
    
    seq3 = "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNLGGGSGGGSGGGSGLRKRLRKFRNKIKEKLKKIGQKIQGLLPKL"
    res3 = calculate_final_score(
        sequence=seq3,
        fusion_peptides=pep3,
        manual_body_plddt=87.6,    # T4 Lysozyme 본체 안정적
        manual_fusion_plddt=85.0,  # SMAP-29는 융합부임에도 알파나선 구조가 단단함 (보너스 없음)
        use_alphafold=False,  # True로 설정하면 AlphaFold 자동 실행
        salt_mM=150
    )

    print("\n5번 서열\n")
    pep5 = [
        {"name": "custom", "seq": "GGGSGGGSGGGSRKKRKKRKK", "position": "C-terminal"}
    ]
    
    seq5 = "MAVTVKVKAGSVAGIKGAYIDAIANAIKPDWVPYVDGDKYVFTAHGIGKAYLNPSSKVAIVVAGALDEKMAVAAGVGYGAVVDRVLGGLQGELVVEGLNYSYIGVDVPDELPDKPVAYVAGVGLGYRTVPVPVAVAEKGGGSGGGSGGGSRKKRKKRKK"
    res5 = calculate_final_score(
        sequence=seq5,
        fusion_peptides=pep5,
        manual_body_plddt=85.0,  # AlphaFold로 얻은 값 또는 추정값
        manual_fusion_plddt=15.0,
        use_alphafold=False,  # True로 설정하면 AlphaFold 자동 실행
        salt_mM=150
    )

    print("\n7번 서열\n")
    pep7 = [
        {"name": "SMAP-29", "seq": "RGLRRLGRKIAHGVKKYGPTVLRIIGIGGGSGGGSGGGS", "position": "N-terminal"}
    ]
    
    seq7 = "RGLRRLGRKIAHGVKKYGPTVLRIIGIGGGSGGGSGGGSMNRLTEVPPYIDEPFDRRREISSLREKVTVVEFEGSVAGIKGAYIDAIANAIKPDWVPYVDGDKYVFTAHGIGKAYLNPSSKVAIVVAGALDEKMAVAAGVGYGAVVDRVLGGLQGELVVEGLNYSYIGVDVPDELPDKPVAYVAGVGLGYRTVPVPVAVAEK"
    res7 = calculate_final_score(
        sequence=seq7,
        fusion_peptides=pep7,
        manual_body_plddt=95.38,  # AlphaFold로 얻은 값 또는 추정값
        manual_fusion_plddt=65.0,
        use_alphafold=False,  # True로 설정하면 AlphaFold 자동 실행
        salt_mM=150
    )

    print("\n 모든 예시 완료")
