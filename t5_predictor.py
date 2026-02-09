"""
T5 Endolysin Activity Predictor
입력: T5 서열 + Fusion peptide
출력: 0-1 스코어 + LNP 진행 권장사항
"""

import os
import warnings
warnings.filterwarnings('ignore')

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


def calculate_structure_score(plddt):
    """
    pLDDT 값을 0-1 스코어로 변환
    """
    # pLDDT 50 이하는 매우 불안정
    # pLDDT 90 이상은 매우 안정적
    score = max(0, min(1, (plddt - 50) / 40))
    return score


# ================== 2. 외막 투과 예측 스코어 ==================

def calculate_peptide_properties(sequence):
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


def calculate_penetration_score(fusion_peptide, fusion_type):
    """
    외막 투과 능력 예측 스코어
    """
    props = calculate_peptide_properties(fusion_peptide)
    
    if props is None:
        return 0.5  # 기본값
    
    # 1. 양전하 점수 (0-1)
    # +3 이상이면 만점
    charge_score = min(1.0, max(0, props['charge']) / 4.0)
    
    # 2. 양친매성 점수 (0-1)
    amphipathicity = calculate_amphipathicity(fusion_peptide)
    amphi_score = min(1.0, amphipathicity / 0.5)  # 0.5 이상이면 만점
    
    # 3. 소수성 비율 점수 (20-40%가 최적)
    hydro_ratio = props['hydrophobic_ratio']
    if 0.2 <= hydro_ratio <= 0.4:
        hydro_score = 1.0
    elif hydro_ratio < 0.2:
        hydro_score = hydro_ratio / 0.2
    else:
        hydro_score = max(0, 1.0 - (hydro_ratio - 0.4) / 0.3)
    
    # Fusion type별 가중치 조정
    if fusion_type in ['KWK', 'kwk']:
        # KWK는 양전하가 중요
        weights = [0.5, 0.2, 0.3]
    elif fusion_type in ['SMAP-29', 'SMAP29', 'smap']:
        # SMAP-29는 양친매성이 중요
        weights = [0.3, 0.5, 0.2]
    elif fusion_type in ['Cys', 'cys', 'CYS']:
        # Cys는 구조적 안정성에 더 의존
        weights = [0.3, 0.3, 0.4]
    else:
        weights = [0.4, 0.3, 0.3]
    
    penetration_score = (
        weights[0] * charge_score +
        weights[1] * amphi_score +
        weights[2] * hydro_score
    )
    
    return penetration_score


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
        model_name = "facebook/esm2_t6_8M_UR50D"
        # 더 정확한 모델 (650M 파라미터) - 느리지만 정확
        # model_name = "facebook/esm2_t33_650M_UR50D"
        
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

def calculate_final_score(sequence, fusion_peptide, fusion_type, 
                          manual_plddt=None, use_alphafold=False):
    """
    최종 0-1 스코어 계산 및 의사결정
    
    Parameters:
    -----------
    sequence : str
        전체 T5 서열 (fusion peptide 포함)
    fusion_peptide : str
        Fusion peptide 서열만
    fusion_type : str
        'KWK', 'SMAP-29', 'Cys' 등
    manual_plddt : float, optional
        수동 입력 pLDDT (AlphaFold 실행 안 할 경우)
    use_alphafold : bool
        True면 AlphaFold2 로컬 실행
    """
    
    print("\n" + "="*60)
    print("T5 Endolysin Activity Prediction")
    print("="*60 + "\n")
    
    results = {}
    
    # 1. 구조 안정성 스코어
    print("📐 1단계: 구조 안정성 분석")
    if use_alphafold:
        plddt = run_alphafold_local(sequence)
    elif manual_plddt is not None:
        plddt = manual_plddt
    else:
        print("⚠️ pLDDT 값을 입력하지 않았습니다. 기본값 75 사용")
        plddt = 75.0
    
    structure_score = calculate_structure_score(plddt)
    results['plddt'] = plddt
    results['structure_score'] = structure_score
    print(f"   평균 pLDDT: {plddt:.1f}")
    print(f"   구조 스코어: {structure_score:.3f}\n")
    
    # 2. 외막 투과 예측 스코어
    print("🧬 2단계: 외막 투과 능력 분석")
    penetration_score = calculate_penetration_score(fusion_peptide, fusion_type)
    results['penetration_score'] = penetration_score
    
    props = calculate_peptide_properties(fusion_peptide)
    if props:
        print(f"   순전하: {props['charge']:+.2f}")
        print(f"   소수성 (GRAVY): {props['gravy']:.3f}")
        print(f"   양전하 비율: {props['positive_ratio']:.1%}")
    print(f"   투과 스코어: {penetration_score:.3f}\n")
    
    # 3. 서열 적합성 스코어
    print("🤖 3단계: AI 서열 적합성 분석")
    sequence_score, mlm_loss = calculate_sequence_score(sequence)
    results['sequence_score'] = sequence_score
    results['mlm_loss'] = mlm_loss
    print(f"   MLM Loss: {mlm_loss:.3f}")
    print(f"   서열 스코어: {sequence_score:.3f}\n")
    
    # 4. 최종 스코어 계산
    print("📊 최종 스코어 계산")
    
    # 가중치
    w_structure = 0.30
    w_penetration = 0.40
    w_sequence = 0.30
    
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
    if final_score >= 0.75:
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
    print("T5 Endolysin Activity Predictor v1.0")
    print("="*60)
    
    # 사용 예시 1: 단일 서열 예측
    print("\n[예시 1] T5-SMAP29 서열 분석\n")
    
    # 실제 서열로 교체하세요
    example_sequence = "MFKLSQRSKDRLVGVHPDLVKVVHRALELTPVDFGITEGVRSLETQKKYVAEGKSKTMKSRHLHGLAVDVVAYPKDKDTWNMKYYRMIADAFKQAGRELGVSVEWGGDWVSFKDGVHFQLPHSKYPDPKLE"
    fusion_peptide = "RKLRRLKRKIAHKVKKY"  # SMAP-29
    
    result = calculate_final_score(
        sequence=example_sequence + fusion_peptide,
        fusion_peptide=fusion_peptide,
        fusion_type="SMAP-29",
        manual_plddt=82.5,  # AlphaFold로 얻은 값 또는 추정값
        use_alphafold=False  # True로 설정하면 AlphaFold 자동 실행
    )
    
    # 사용 예시 2: 배치 예측 (주석 해제하여 사용)
    # batch_prediction("sequences.csv", "predictions.csv")
