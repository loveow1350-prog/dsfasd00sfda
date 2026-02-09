# 테스트 디렉토리

시스템 검증 및 테스트 스크립트 모음

## 파일 목록

### validate_system.py
시스템 환경 및 의존성 검증
```bash
python tests/validate_system.py
```

### test_pipeline.py
전체 파이프라인 테스트 (pytest)
```bash
pytest tests/test_pipeline.py -v
```

### quick_start.py
빠른 시작 및 간단한 테스트
```bash
python tests/quick_start.py
```

## 사용법

**중요**: 반드시 프로젝트 루트 디렉토리에서 실행하세요!

```bash
# 프로젝트 루트로 이동
cd /mnt/c/Users/pegoo/Desktop/nlp_project_2
# 또는 Windows에서: cd C:\Users\pegoo\Desktop\nlp_project_2

# 시스템 검증
python tests/validate_system.py

# 빠른 테스트
python tests/quick_start.py

# 전체 파이프라인 테스트
pytest tests/test_pipeline.py -v
```

## 주의사항

- ✅ **올바른 실행 위치**: 프로젝트 루트 (`nlp_project_2/`)
- ❌ **잘못된 실행 위치**: `tests/` 디렉토리 안에서 실행하면 모듈 import 오류 발생

## PDF 파일 준비

테스트 실행 전 PDF 파일을 프로젝트 루트에 배치하세요:
```bash
# 현재 위치 확인
pwd
# /mnt/c/Users/pegoo/Desktop/nlp_project_2 여야 함

# PDF 파일 확인
ls 중간보고서_자연어처리.pdf
```
