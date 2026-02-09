"""
System validation script - Check if all components are ready
"""
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def check_module_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description}: {module_name} (설치 필요)")
        return False


def check_env_file():
    """Check .env file"""
    env_exists = Path('.env').exists()
    if env_exists:
        print("✅ .env 파일 존재")

        # Check for API keys
        with open('.env', 'r') as f:
            content = f.read()

        has_openai = 'OPENAI_API_KEY=' in content and 'your_' not in content
        has_anthropic = 'ANTHROPIC_API_KEY=' in content and 'your_' not in content

        if has_openai or has_anthropic:
            print("  ✅ LLM API 키 설정됨")
        else:
            print("  ⚠️  LLM API 키 미설정 (실행 시 오류 발생)")

        return True
    else:
        print("❌ .env 파일 없음")
        print("  실행: cp .env.example .env")
        return False


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("Multi-Agent Pipeline 시스템 검증")
    print("=" * 60)

    all_checks = []

    # Check core files
    print("\n📁 필수 파일 검사:")
    all_checks.append(check_file_exists("config/config.yaml", "설정 파일"))
    all_checks.append(check_file_exists("config/prompts.yaml", "프롬프트 파일"))
    all_checks.append(check_file_exists("requirements.txt", "의존성 파일"))

    # Check Python files
    print("\n🐍 Python 모듈 검사:")
    all_checks.append(check_file_exists("src/models.py", "데이터 모델"))
    all_checks.append(check_file_exists("src/utils.py", "유틸리티"))
    all_checks.append(check_file_exists("src/pdf_extractor.py", "PDF 추출기"))
    all_checks.append(check_file_exists("src/structure_parser.py", "구조 파서"))
    all_checks.append(check_file_exists("src/step_decomposer.py", "스텝 분해기"))
    all_checks.append(check_file_exists("src/search_client.py", "검색 클라이언트"))
    all_checks.append(check_file_exists("src/problem_analyzer.py", "문제 분석기"))
    all_checks.append(check_file_exists("main_pipeline.py", "메인 파이프라인"))

    # Check dependencies
    print("\n📦 필수 의존성 검사:")
    deps_check = [
        check_module_import("fitz", "PyMuPDF"),
        check_module_import("pydantic", "Pydantic"),
        check_module_import("yaml", "PyYAML"),
    ]
    all_checks.extend(deps_check)

    # Check optional dependencies
    print("\n📦 선택 의존성 검사 (없어도 작동):")
    check_module_import("redis", "Redis")
    check_module_import("openai", "OpenAI SDK")
    check_module_import("anthropic", "Anthropic SDK")
    check_module_import("duckduckgo_search", "DuckDuckGo Search")

    # Check environment
    print("\n🔑 환경 설정 검사:")
    env_check = check_env_file()

    # Check input file
    print("\n📄 입력 파일 검사:")
    pdf_exists = check_file_exists("중간보고서_자연어처리.pdf", "입력 PDF")

    # Summary
    print("\n" + "=" * 60)
    print("검증 결과")
    print("=" * 60)

    required_count = sum(all_checks)
    total_required = len(all_checks)

    if required_count == total_required and env_check and pdf_exists:
        print("✅ 모든 검사 통과! 시스템 실행 준비 완료")
        print("\n다음 명령으로 실행하세요:")
        print("  python main_pipeline.py 중간보고서_자연어처리.pdf")
        return 0

    else:
        print(f"⚠️  {total_required - required_count}개의 필수 항목 누락")
        print("\n필요한 조치:")

        if required_count < total_required:
            print("  1. 누락된 의존성 설치:")
            print("     pip install -r requirements.txt")

        if not env_check:
            print("  2. 환경 설정 파일 생성:")
            print("     cp .env.example .env")
            print("     (그 후 .env 파일에서 API 키 설정)")

        if not pdf_exists:
            print("  3. 입력 PDF 파일 준비:")
            print("     중간보고서_자연어처리.pdf를 현재 디렉토리에 배치")

        return 1


if __name__ == "__main__":
    sys.exit(main())
