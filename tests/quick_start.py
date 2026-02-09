"""
Quick start script - Test PDF extraction without full dependencies
"""
from pathlib import Path

def test_basic_extraction():
    """Test basic PDF extraction without LLM"""
    try:
        import fitz  # PyMuPDF

        pdf_path = "sample/중간보고서_자연어처리.pdf"

        if not Path(pdf_path).exists():
            print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return

        print(f"📄 PDF 파일 열기: {pdf_path}")
        doc = fitz.open(pdf_path)

        print(f"✅ 총 페이지 수: {len(doc)} 페이지")
        print(f"📊 메타데이터: {doc.metadata.get('title', 'N/A')}")

        # Extract first page text
        if len(doc) > 0:
            page = doc[0]
            text = page.get_text()

            print(f"\n📝 첫 페이지 텍스트 미리보기 (200자):")
            print("-" * 50)
            print(text[:200])
            print("-" * 50)

            # Get text with formatting
            blocks = page.get_text("dict")
            block_count = len(blocks.get("blocks", []))
            print(f"\n📦 첫 페이지 텍스트 블록 수: {block_count}")

        doc.close()

        print("\n✅ PDF 추출 테스트 성공!")
        print("\n다음 단계:")
        print("1. .env.example을 .env로 복사하고 API 키 설정")
        print("2. pip install -r requirements.txt 실행")
        print("3. python main_pipeline.py sample/중간보고서_자연어처리.pdf 실행")

    except ImportError:
        print("❌ PyMuPDF가 설치되지 않았습니다.")
        print("실행: pip install PyMuPDF")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    test_basic_extraction()
