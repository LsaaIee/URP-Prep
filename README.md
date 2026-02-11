<div align="center">

# 🧬 T5 Endolysin Activity Predictor

### URP 프로젝트 - 서열 기반 활성 예측 시스템

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**AI 기반 실험 가속화: 서열 입력 → 스코어 계산 → 다음 단계 자동 추천**

[빠른 시작](#-빠른-시작) • [사용법](#-사용법) • [기능](#-주요-기능) • [설치](#-설치-가이드)

</div>

---

## 📋 목차

- [개요](#-개요)
- [주요 기능](#-주요-기능)
- [설치 가이드](#-설치-가이드)
- [사용법](#-사용법)
  - [CLI 버전](#1-cli-버전-명령줄)
  - [웹 인터페이스](#2-웹-인터페이스-streamlit)
- [AlphaFold2 pLDDT 추출](#-alphafold2-plddt-추출)
- [입출력 형식](#-입출력-형식)
- [문제 해결](#-문제-해결)

---

## 🎯 개요

T5 endolysin 변형 서열의 활성을 예측하여 실험 우선순위를 결정하는 AI 기반 도구입니다.

### 해결하는 문제
- ⏰ **시간 절약**: 수백 개의 후보 서열 중 유망한 것만 실험
- 🎯 **정확한 의사결정**: 0-1 스코어 기반 LNP 진행 여부 판단
- 🤖 **AI 분석**: AlphaFold2, ESM-2, 물리화학적 특성 종합 평가

### 예측 기준
| 스코어 | 의사결정 | 예상 효과 |
|--------|---------|----------|
| **≥0.75** | ✅ LNP 단계 진행 | CFU reduction ≥99% |
| **0.60-0.74** | ⚠️ CFU 2-3회 테스트 | CFU reduction 90-98% |
| **0.45-0.59** | ⚡ 조건 최적화 필요 | CFU reduction 70-90% |
| **<0.45** | ❌ 서열 재설계 | CFU reduction <70% |

---

## ✨ 주요 기능

### 🔬 3가지 분석 모듈

<table>
<tr>
<td width="33%">

**🏗️ 구조 안정성**
- AlphaFold2 pLDDT 기반
- 가중치: 30%
- 단백질 접힘 신뢰도

</td>
<td width="33%">

**🧪 외막 투과 능력**
- 전하, 소수성, 양친매성
- 가중치: 40%
- KWK/SMAP-29/Cys 분석

</td>
<td width="33%">

**🤖 서열 적합성**
- ESM-2 언어 모델
- 가중치: 30%
- 자연스러운 서열 여부

</td>
</tr>
</table>

### 💡 지원 기능
- ✅ 단일 서열 실시간 예측
- ✅ 배치 예측 (CSV 업로드)
- ✅ 웹 기반 인터페이스
- ✅ 상세 분석 리포트
- ✅ 결과 CSV 다운로드

---

## 🚀 빠른 시작

```bash
# 1. 저장소 다운로드
git clone https://github.com/your-repo/urp-t5-predictor.git
cd urp-t5-predictor

# 2. 라이브러리 설치
pip install biopython pandas numpy streamlit torch transformers fair-esm

# 3. 웹 인터페이스 실행
streamlit run streamlit_app.py
```

> 💻 **브라우저가 자동으로 열립니다** → `http://localhost:8501`

---

## 📦 설치 가이드

### 필수 요구사항
- Python 3.8 이상
- 8GB RAM 이상 권장
- 인터넷 연결 (모델 다운로드용)

### Step 1: Python 패키지 설치

```bash
# 기본 패키지
pip install biopython pandas numpy streamlit

# 딥러닝 패키지
pip install torch transformers

# ESM-2 모델
pip install fair-esm
```

<details>
<summary>📌 설치 문제 해결 (클릭하여 펼치기)</summary>

**PyTorch 설치 실패 시:**
```bash
# CPU 버전 (가벼움)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU 버전 (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**M1/M2 Mac 사용자:**
```bash
# Conda 환경 사용 권장
conda create -n t5_pred python=3.10
conda activate t5_pred
conda install pytorch -c pytorch
pip install biopython pandas streamlit transformers fair-esm
```

</details>

---

## 📖 사용법

### 1. CLI 버전 (명령줄)

#### 단일 서열 예측

1. `t5_predictor.py` 파일 열기
2. **450-451번 줄** 수정:

```python
# 예시: T5-SMAP29 서열
example_sequence = "MKTLIALSAALAAAGSAHARQPQQRDGCPGHMLRYVNINHYRWVYGKVSRKGPLYANYSRGGYTYTPPPRTPNASNTEISPLPPGF"
fusion_peptide = "RGLRRLGRKIAHGVKKYGPTVLRIIRIA"  # SMAP-29
```

3. 실행:

```bash
python t5_predictor.py
```

4. 출력 예시:

```
============================================================
T5 Endolysin Activity Prediction
============================================================

📐 1단계: 구조 안정성 분석
   평균 pLDDT: 82.5
   구조 스코어: 0.650

🧬 2단계: 외막 투과 능력 분석
   순전하: +9.10
   소수성 (GRAVY): -0.20
   투과 스코어: 0.720

🤖 3단계: AI 서열 적합성 분석
   MLM Loss: 2.83
   서열 스코어: 0.414

📊 최종 스코어: 0.648

💡 권장사항: ⚠️ TEST 2-3 CFUs
   중간 수준의 활성이 예측됩니다.
   예상 효과: 90-98% CFU reduction
```

#### 배치 예측 (여러 서열)

1. `sequences.csv` 파일 생성:

```csv
name,sequence,fusion_peptide,fusion_type,plddt
T5-Cys-001,MKTLI...CYS,CYS,Cys,85.2
T5-KWK-002,MKTLI...KWKKWK,KWKKWK,KWK,78.5
T5-SMAP-003,MKTLI...RGLRRLGRKIAHGVKKY,RGLRRLGRKIAHGVKKY,SMAP-29,82.3
```

2. 코드에서 주석 해제 (450번 줄 근처):

```python
# 배치 예측 실행
batch_prediction("sequences.csv", "predictions.csv")
```

3. 실행 후 `predictions.csv` 확인

---

### 2. 웹 인터페이스 (Streamlit)

#### 실행

```bash
streamlit run streamlit_app.py
```

#### 사용 화면

<table>
<tr>
<td width="50%">

**📝 입력 패널**
- 전체 T5 서열 (FASTA 형식)
- Fusion peptide 종류 선택
  - KWK
  - SMAP-29
  - Cys
  - Custom
- Fusion peptide 서열
- 위치 (C/N-terminal)
- pLDDT 값 입력

</td>
<td width="50%">

**📊 결과 패널**
- 최종 스코어 (0-1)
- 3가지 세부 스코어
- 의사결정 권장사항
- 상세 분석 결과
- CSV 다운로드

</td>
</tr>
</table>

#### 단계별 가이드

```
1. 서열 입력
   ┌────────────────────────────────┐
   │ >T5-SMAP29                     │
   │ MKTLIFGKLLKKALKFLGKVLK...      │
   └────────────────────────────────┘

2. Fusion Peptide 설정
   종류: [SMAP-29 ▼]
   서열: RGLRRLGRKIAHGVKKYGPTVLRIIRIA
   위치: [C-terminal ▼]

3. pLDDT 입력
   평균 pLDDT: [82.5]

4. 🚀 예측 시작 클릭

5. 결과 확인 및 다운로드
```

---

## 📋 입출력 형식

### 입력 형식

#### FASTA 형식 (권장)
```
>T5_KWK_001
MKTLIALSAALAAAGSAHARQPQQRDGCPGHMLRYVNINHYRWVKWKKWK
```

#### Plain 서열
```
MKTLIALSAALAAAGSAHARQPQQRDGCPGHMLRYVNINHYRWVKWKKWK
```

### 출력 형식 (CSV)

```csv
name,final_score,structure_score,penetration_score,sequence_score,plddt,mlm_loss,decision,expected_cfu
T5-SMAP-001,0.648,0.650,0.720,0.414,82.5,2.83,TEST 2-3 CFUs,90-98% CFU reduction
```

---

## ⚙️ 고급 설정

### 가중치 조정

`t5_predictor.py` **376-378번 줄** 수정:

```python
# 기본 가중치
w_structure = 0.30    # 구조 안정성
w_penetration = 0.40  # 외막 투과성
w_sequence = 0.30     # 서열 적합성

# 예시: 투과성을 더 중요하게
w_structure = 0.25
w_penetration = 0.50
w_sequence = 0.25
```

### Threshold 조정

**387-400번 줄** 수정:

```python
if final_score >= 0.75:
    decision = "✅ PROCEED TO LNP"
elif final_score >= 0.60:
    decision = "⚠️ TEST 2-3 CFUs"
# ...
```

---

## 🐛 문제 해결

<details>
<summary><b>ImportError: No module named 'transformers'</b></summary>

```bash
pip install --upgrade transformers
```

모델 다운로드 실패 시:
```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install transformers
```

</details>

<details>
<summary><b>OutOfMemoryError (ESM-2)</b></summary>

**작은 모델 사용** (`t5_predictor.py` 245번 줄):

```python
# 기존
model_name = "facebook/esm2_t33_650M_UR50D"

# 변경 (메모리 절약)
model_name = "facebook/esm2_t6_8M_UR50D"
```

</details>

<details>
<summary><b>Streamlit 실행 안 됨</b></summary>

1. 포트 충돌 확인:
```bash
streamlit run streamlit_app.py --server.port 8502
```

2. 브라우저 수동 접속:
```
http://localhost:8501
```

</details>

<details>
<summary><b>AlphaFold Colab 느림</b></summary>

**GPU 사용 확인**:
- 런타임 → 런타임 유형 변경 → GPU 선택

무료 GPU 한도 초과 시:
- 다음 날 재시도
- 또는 수동 pLDDT 입력 (기본값 75)

</details>

---

## 📚 참고 자료

### 사용된 기술
- **AlphaFold2**: 구조 예측 ([Paper](https://www.nature.com/articles/s41586-021-03819-2))
- **ESM-2**: 단백질 언어 모델 ([Paper](https://www.science.org/doi/10.1126/science.ade2574))
- **Biopython**: 생물정보학 툴킷 ([Docs](https://biopython.org/))

### 프로젝트 배경
- URP 프로젝트: Endolysin 기반 항균제 개발
- 목표: T5 변형체 활성 예측으로 실험 효율화

---

## 📞 문의 및 기여

### 버그 리포트
Issue 탭에서 보고해주세요.

### 기여 방법
Pull Request를 환영합니다!

---

<div align="center">

**Made with ❤️ for URP Project**

⭐ 도움이 되었다면 Star를 눌러주세요!

</div>
