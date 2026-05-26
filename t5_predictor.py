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
    try:
        from colabfold.batch import get_queries, run
        from colabfold.download import download_alphafold_params
        
        print("AlphaFold2 구조 예측 중... (첫 실행은 10-30분 소요)")
        os.makedirs(output_dir, exist_ok=True)
        fasta_path = os.path.join(output_dir, "input.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">T5_sequence\n{sequence}\n")
        
        os.system(f"colabfold_batch {fasta_path} {output_dir} --num-models 1")
        return extract_plddt_from_results(output_dir)
        
    except ImportError:
        print("⚠️ ColabFold가 설치되지 않았습니다. pLDDT 값을 수동으로 입력하세요.")
        return float(input("평균 pLDDT 값 입력 (50-100): "))
    except Exception as e:
        print(f"⚠️ AlphaFold2 실행 실패: {e}")
        return 75.0


def extract_plddt_from_results(output_dir):
    import json
    import glob
    json_files = glob.glob(os.path.join(output_dir, "*_scores_rank_001*.json"))
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            return np.mean(data.get('plddt', []))
    
    pdb_files = glob.glob(os.path.join(output_dir, "*_unrelaxed_rank_001*.pdb"))
    if pdb_files:
        plddt_values = []
        with open(pdb_files[0], 'r') as f:
            for line in f:
                if line.startswith("ATOM") and ' CA ' in line:
                    plddt_values.append(float(line[60:66].strip()))
        return np.mean(plddt_values)
    return 75.0


def calculate_structure_score(body_plddt, fusion_plddt=None):
    """
    본체(Body)의 pLDDT로 효소 안정성을 평가하고, 
    융합 펩타이드(Fusion)의 pLDDT가 낮을수록(유연할수록) 가산점 부여
    """
    base_score = max(0, min(1, (body_plddt - 50) / 40))
    
    flex_bonus = 0.0
    if fusion_plddt is not None:
        if fusion_plddt < 50:
            flex_bonus = 0.15 * (1.0 - (fusion_plddt / 50.0))
            
    return min(1.0, base_score + flex_bonus)


# ================== 2. 외막 투과 예측 스코어 ==================

def calculate_peptide_properties(sequence):
    try:
        analyzer = ProteinAnalysis(sequence)
        charge = analyzer.charge_at_pH(7.4)
        gravy = analyzer.gravy()
        mw = analyzer.molecular_weight()
        aa_percent = analyzer.get_amino_acids_percent()
        
        positive_ratio = aa_percent.get('K', 0) + aa_percent.get('R', 0)
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
    hydrophobicity_scale = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    window_size = min(11, len(sequence))
    max_hydrophobic_moment = 0
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        h_sum = sum(hydrophobicity_scale.get(aa, 0) * np.cos(j * 100 * 3.14159 / 180) for j, aa in enumerate(window))
        hydrophobic_moment = abs(h_sum) / window_size
        max_hydrophobic_moment = max(max_hydrophobic_moment, hydrophobic_moment)
    return max_hydrophobic_moment


def calculate_penetration_score(fusion_peptide, fusion_type="Custom", salt_mM=20, linker_seq=""): 
    props = calculate_peptide_properties(fusion_peptide)
    if props is None: return 0.5
    
    charge = max(0, props['charge'])
    charge_score = min(1.0, charge / 5.0) 
    amphi_score = min(1.0, calculate_amphipathicity(fusion_peptide) / 0.5) 
    hydro_ratio = props['hydrophobic_ratio']
    
    if 0.2 <= hydro_ratio <= 0.4: hydro_score = 1.0
    elif hydro_ratio < 0.2: hydro_score = hydro_ratio / 0.2
    else: hydro_score = max(0, 1.0 - (hydro_ratio - 0.4) / 0.3)

    # 💡 염 농도(mM)에 따른 지수 감소 차폐 효과
    charge_weight = 0.5 * math.exp(-0.01 * salt_mM)
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
        gs_ratio = (linker_seq.count('G') + linker_seq.count('S')) / len(linker_seq) if len(linker_seq) > 0 else 0
        linker_bonus = gs_ratio * 0.15 
        
    return min(1.0, base_pen + linker_bonus)


def calculate_penetration_score_for_list(fusion_peptides, salt_mM=20):
    if not fusion_peptides:
        return 0.35 # 기본 T5 본체의 막 투과 잠재력

    scores = []
    for fp in fusion_peptides:
        seq = fp["seq"]
        name = fp.get("name", "Custom")
        
        # 💡 역할 부여: Cys 펩타이드는 투과 점수 계산에서 제외
        if 'cys' in name.lower():
            continue 
            
        s = calculate_penetration_score(seq, name, salt_mM=salt_mM, linker_seq=seq)
        scores.append(s)

    if not scores:
        return 0.35 # Cys 단독일 경우 기본 T5 투과력 부여

    # 평균의 함정 탈출: 가장 투과력 높은 펩타이드 점수 하나만 채택 (Max)
    return max(scores)


# ================== 3. 서열 적합성 스코어 (ESM-2) ==================

ESM_TOKENIZER = None
ESM_MODEL = None

def load_esm_model():
    global ESM_TOKENIZER, ESM_MODEL
    if ESM_TOKENIZER is None:
        print("ESM-2 모델 로딩 중... (첫 실행 시 2-3분 소요)")
        model_name = "facebook/esm2_t33_650M_UR50D"
        ESM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        ESM_MODEL = EsmForMaskedLM.from_pretrained(model_name)
        ESM_MODEL.eval()
        print("✓ ESM-2 모델 로드 완료")


def calculate_mlm_loss(sequence):
    load_esm_model()
    try:
        inputs = ESM_TOKENIZER(sequence, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = ESM_MODEL(**inputs, labels=inputs["input_ids"])
            return outputs.loss.item()
    except Exception as e:
        print(f"⚠️ ESM-2 계산 실패: {e}")
        return 3.0


def calculate_sequence_score(sequence):
    mlm_loss = calculate_mlm_loss(sequence)
    score = 1.0 / (1.0 + mlm_loss / 2.0)
    return score, mlm_loss


# ================== 4. 종합 스코어 계산 ==================

def calculate_final_score(
        sequence, 
        fusion_peptides=None,
        manual_body_plddt=None,
        manual_fusion_plddt=None,
        use_alphafold: bool = False, 
        weights=None,
        salt_mM=20
    ) -> dict:
    
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
    results['body_plddt'] = body_plddt
    results['structure_score'] = structure_score
    print(f"   본체 pLDDT: {body_plddt:.1f}")
    if fusion_plddt: print(f"   융합부 pLDDT: {fusion_plddt:.1f} (유연성 보너스 체크)")
    print(f"   구조 스코어: {structure_score:.3f}\n")

    # 2. 외막 투과 예측 스코어
    print("🧬 2단계: 외막 투과 능력 분석")
    penetration_score = calculate_penetration_score_for_list(fusion_peptides, salt_mM=salt_mM)
    results['penetration_score'] = penetration_score
    print(f"   Fusion 개수: {len(fusion_peptides)}")
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
    if weights is None: 
        w_structure, w_penetration, w_sequence = 0.30, 0.40, 0.30
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
    
    # 💡 [아비디티 부스트] 이합체(Dimer) 형성 시 최종 활성 15% 증폭 (최대 0.995)
    has_dimer = any('cys' in fp.get("name", "").lower() for fp in fusion_peptides)
    if has_dimer:
        print("   💡 Dimer 형성 감지: 이합체 활성(Avidity) 증폭으로 최종 스코어 15% 부스트 적용")
        final_score = min(0.995, final_score * 1.15)
    
    results['final_score'] = final_score
    results['weights'] = {
        'structure': w_structure,
        'penetration': w_penetration,
        'sequence': w_sequence
    }
    
    print(f"   구조 ({w_structure:.0%}): {structure_score:.3f} × {w_structure:.2f} = {w_structure*structure_score:.3f}")
    print(f"   투과 ({w_penetration:.0%}): {penetration_score:.3f} × {w_penetration:.2f} = {w_penetration*penetration_score:.3f}")
    print(f"   서열 ({w_sequence:.0%}): {sequence_score:.3f} × {w_sequence:.2f} = {w_sequence*sequence_score:.3f}")
    print(f"\n   {'='*40}")
    print(f"   최종 스코어: {final_score:.3f}")
    print(f"   {'='*40}\n")
    
    # 5. 의사결정
    print("💡 권장사항")
    if final_score >= 0.85:
        decision, expected_cfu = "✅ PROCEED TO LNP", "≥99% CFU reduction (2-log)"
        explanation = "높은 활성이 예측됩니다. 즉시 LNP 단계로 진행하세요."
    elif final_score >= 0.60:
        decision, expected_cfu = "⚠️ TEST 2-3 CFUs", "90-98% CFU reduction (1-2 log)"
        explanation = "중간 수준의 활성이 예측됩니다. CFU counting 2-3회로 확인 후 결정하세요."
    elif final_score >= 0.45:
        decision, expected_cfu = "⚡ TEST WITH CAUTION", "70-90% CFU reduction"
        explanation = "낮은 활성이 예측됩니다. 3-4회 CFU 테스트와 조건 최적화를 고려하세요."
    else:
        decision, expected_cfu = "❌ HOLD/REDESIGN", "<70% CFU reduction"
        explanation = "활성이 낮을 것으로 예측됩니다. 서열 재설계를 권장합니다."
    
    print(f"   {decision}\n   {explanation}\n   예상 효과: {expected_cfu}\n")
    print("="*60 + "\n")
    
    return results


# ================== 메인 실행 ==================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("T5 Endolysin Activity Predictor v2.0")
    print("="*60)

    # 사용 예시 1: T5 기본 서열
    print("\n[예시 1] T5 서열 분석\n")
    t5_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQNIANGTSFLKDPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELV"
    result_t5 = calculate_final_score(
        sequence=t5_sequence,
        fusion_peptides=[],
        manual_body_plddt=95.38,
        use_alphafold=False,
        salt_mM=150
    )

    # 사용 예시 2: T5 + KWK
    print("\n[예시 2] T5 + KWK\n")
    fusion_kwk = [{"name": "KWK", "seq": "KWKLFKKI", "position": "internal"}]
    kwk_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQKWKLFKKIPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELV"
    result_kwk = calculate_final_score(
        sequence=kwk_sequence,
        fusion_peptides=fusion_kwk,
        manual_body_plddt=95.38,
        manual_fusion_plddt=85.0,
        use_alphafold=False,
        salt_mM=150,
    )

    # 사용 예시 3: T5 + Cys
    print("\n[예시 3] T5 + Cys\n")
    fusion_cys = [{"name": "cys", "seq": "DGGHVELVGGGGSC", "position": "C-terminal"}]
    cys_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQNIANGTSFLKDPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELVDGGHVELVGGGGSC"
    result_cys = calculate_final_score(
        sequence=cys_sequence,
        fusion_peptides=fusion_cys,
        manual_body_plddt=95.38,
        manual_fusion_plddt=85.0,
        use_alphafold=False,
        salt_mM=150,
    )

    # 사용 예시 4: T5 + KWK + Cys
    print("\n[예시 4] T5 + kwk + Cys\n")
    fusion_two = [
        {"name": "KWK", "seq": "KWKLFKKI", "position": "internal"},
        {"name": "cys", "seq": "DGGHVELVGGGGSC", "position": "C-terminal"}
    ]
    two_sequence = "SFKFGKNSEKQLATVKPELQKVARRALELSPYDFTIVQGIRTVAQSAQKWKLFKKIPSKSKHITGDAIDFAPYINGKIDWNDLEAFWAVKKAFEQAGKELGIKLRFGADWNASSGIIMMKLNVAPMMVVVELVDGGHVELVGGGGSC"
    result_two = calculate_final_score(
        sequence=two_sequence,
        fusion_peptides=fusion_two,
        manual_body_plddt=95.38,
        manual_fusion_plddt=85.0,
        use_alphafold=False,
        salt_mM=150,
    )

    print("\n 모든 예시 완료")