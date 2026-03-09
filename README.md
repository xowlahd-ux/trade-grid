# keundon_grid_repo

GitHub Actions 기반 병렬 그리드 레포입니다.

## 목적
- 메타 엔진 그리드
- 브랜치5A 엔진 그리드
- 실행모드 반영 비교
  - mode1: D일 판단 -> D+1 풀 리밸런싱
  - mode2: D일 판단 -> D+1 매도 -> D+2 매수

## 핵심 결과물
각 조합마다 아래 지표를 CSV로 저장합니다.
- CAGR
- MDD
- 최대 회복기간
- `selection_score`
  - `cagr - 0.50 * abs(mdd) - 0.15 * (max_recovery_days / 252)`
- 최근 10년 CAGR / MDD / 최대 회복기간 / selection_score
- 사용한 파라미터 전부 (`param::...` 컬럼)

## 사용 workflow
- `.github/workflows/grid-parallel.yml`

## 기본 입력
- start_date
- end_date (`latest` 지원)
- execution_mode
- buy_cost
- sell_cost
- shard_count (동적 매트릭스, 1~64)
- meta_grid_yaml
- branch_grid_yaml
- hybrid_core_weight
- hybrid_satellite_weight

## 개선 사항
- meta 월말/분기말 리밸런싱을 **마지막 거래일 기준**으로 수정
- workflow의 `shard_count`를 **실제 동적 matrix**로 반영
- meta 내부 수수료와 실행 수수료를 **workflow 입력 기준으로 일치**
- hybrid 결과 폴더/엔진명을 **실제 비중 기반으로 동적 생성**
- `meta_best`, `branch5a_best`, hybrid best 폴더를 **`out/aggregate/` 아래로 저장**
- `end_date=latest` 별칭 추가
- grid 확장기가 **list-of-dicts 프로필형 옵션**을 지원
- meta / branch / hybrid 타깃에 **vol targeting** 적용 가능
- aggregate best 선택 기준을 **selection_score 중심**으로 조정

## 아티팩트
- shard별 결과 CSV/JSON
- `out/aggregate/all_results.csv`
- `out/aggregate/top50_by_score.csv`
- `out/aggregate/top50_by_cagr.csv`
- `out/aggregate/top50_by_mdd.csv`
- `out/aggregate/summary.csv`
- `out/aggregate/meta_best/`
- `out/aggregate/branch5a_best/`
- `out/aggregate/hybrid_*/`
