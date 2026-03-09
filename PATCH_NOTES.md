# Patch Notes

## 이번에 추가/수정한 내용
- meta 월말/분기말 리밸런싱을 **캘린더 말일이 아니라 마지막 거래일** 기준으로 유지
- `shard_count`가 실제 shard 개수와 일치하도록 **동적 matrix workflow** 유지
- meta 내부 비용 계산을 workflow 입력 `buy_cost` / `sell_cost`와 **동일하게 강제** 유지
- hybrid 산출물 이름을 실제 비중에 맞춰 `hybrid_<core>_<satellite>`로 유지
- `out/aggregate/` 아래에 `meta_best`, `branch5a_best`, hybrid best를 저장하는 구조 유지
- grid 확장기가 **list-of-dicts 옵션**을 지원하도록 확장해서, `allocator`, `asset_crash`, `vol_targeting` 같은 설정을 프로필 단위로 탐색 가능하게 변경
- `aggregate_results.py`의 best 선택 기준을 **CAGR-only에서 balanced score** 기반으로 변경
  - `selection_score = cagr - 0.50 * abs(mdd) - 0.15 * (max_recovery_days / 252)`
- meta / branch / hybrid target에 공통으로 적용되는 **vol targeting 모듈** 추가
  - 최근 실현 변동성이 목표치를 넘으면 risky sleeve를 축소
  - 남는 비중은 `SGOV_MIX`로 이동
  - lookback / target annual vol / min scale를 config에서 제어
- `config/grid_meta.yml`을 추천 방향으로 재설계
  - `weekly / biweekly / monthly` 리밸런싱 탐색
  - `top_n=2` 포함
  - bull 구간 defensive sleeve 허용 프로필 추가
  - `QQQ / SPY` asset crash 보호 프로필 추가
  - stricter `sgov_exit_assist` 추가
  - `recovery_boost` 정리
  - `vol_targeting` 프로필 추가
- `config/grid_branch5a.yml`을 확장
  - lookback 7개
  - `weekly / biweekly / monthly`
  - `top1_weight` 0.55 / 0.60 / 0.65
  - `vol_targeting` 프로필 추가
- `config/final_meta_fixed.yml`도 추천형 단일 설정으로 업데이트

## 그리드 크기
- `config/grid_meta.yml`: **6,912 combos**
- `config/grid_branch5a.yml`: **189 combos**

## 확인한 것
- 합성 가격 데이터 기준으로 아래 경로 smoke test 통과
  - `scripts/run_grid_shard.py --engine meta`
  - `scripts/run_grid_shard.py --engine branch5a`
  - `scripts/aggregate_results.py`
- vol targeting 활성 조합에서 실제로
  - `vt_binding` 발생
  - `vt_scale`가 1.0 아래로 내려감
  - 축소된 비중이 `SGOV_MIX`로 이동하는 것 확인
- smoke test 결과물에서 아래를 확인
  - `selection_score` 컬럼 생성
  - `top50_by_score.csv` 생성
  - `meta_best/summary.csv`, `branch5a_best/summary.csv`, `hybrid_70_30/summary.csv` 생성
