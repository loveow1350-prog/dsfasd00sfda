import os
import json
import tempfile
from pathlib import Path
import sys
import streamlit as st

# ✅ 프로젝트 코드 import (레포 루트에서 실행 기준)
from main_pipeline import PipelineOrchestrator
from src.evaluation_orchestrator import EvaluationOrchestrator
from src.evaluation_report_generator import ReportGenerator

# ✅ Markdown -> PDF 변환 (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.units import mm

# ✅ 한글 폰트 등록(ReportLab)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import asyncio

# Windows에서 적절한 이벤트 루프 정책을 설정
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---- PDF 한글 폰트 등록 ----
FONT_PATH = "C:/Users/boows/CheckPoint-AI/fonts/NotoSansKR-VariableFont_wght.ttf"
pdfmetrics.registerFont(TTFont("NotoSansKR", FONT_PATH))
# ---------------------------


def md_inline_bold_to_html(text: str) -> str:
    """
    **bold** → <b>bold</b> 변환 (문장 중간 포함 전부 처리)
    """
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def md_to_pdf(md_text: str, out_path: str, title: str = "Report"):
    """
    아주 가벼운 Markdown -> PDF 변환기 (heading/bullet/codeblock 중심)
    - 완전한 markdown 렌더러는 아니지만, '보고서용'으로 충분히 읽히게 만듭니다.
    - **bold** 인라인 볼드 처리 지원
    - 한글 폰트 등록 시 한글 출력 지원
    """
    font_ok = register_korean_font()

    styles = getSampleStyleSheet()

    # fontName 지정 (등록 성공 시)
    base_font = FONT_NAME if font_ok else styles["BodyText"].fontName
    base_bold_font = base_font  # reportlab은 <b> 태그로 굵기 처리(폰트가 bold variant 없으면 굵기 효과 제한)

    h1 = ParagraphStyle(
        "h1",
        parent=styles["Heading1"],
        spaceAfter=10,
        fontName=base_font,
    )
    h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        spaceAfter=8,
        fontName=base_font,
    )
    h3 = ParagraphStyle(
        "h3",
        parent=styles["Heading3"],
        spaceAfter=6,
        fontName=base_font,
    )
    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=6,
        fontName=base_font,
    )
    mono = ParagraphStyle(
        "mono",
        parent=styles["Code"],
        leading=12,
        spaceAfter=6,
        fontName=base_font,  # 코드도 한글 깨지면 동일 폰트 사용
    )

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    story = [Paragraph(title, h1), Spacer(1, 8)]

    in_code = False
    code_lines = []

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip("\n")

        # code fence
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                story.append(Preformatted("\n".join(code_lines), mono))
                story.append(Spacer(1, 6))
            continue

        if in_code:
            code_lines.append(line)
            continue

        # headings
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), h1))
            story.append(Spacer(1, 6))
            continue
        if line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), h2))
            story.append(Spacer(1, 4))
            continue
        if line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), h3))
            story.append(Spacer(1, 4))
            continue

        # bullet
        if line.strip().startswith("- "):
            bullet_text = line.strip()[2:]

            # HTML-safe escape 먼저
            safe = (
                bullet_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            # **bold** 처리
            safe = md_inline_bold_to_html(safe)

            story.append(Paragraph("• " + safe, body))
            continue

        # horizontal rule
        if line.strip() in ("---", "***"):
            story.append(Spacer(1, 10))
            continue

        # blank
        if not line.strip():
            story.append(Spacer(1, 6))
            continue

        # normal paragraph
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # **bold** 처리 (문장 중간 포함)
        safe = md_inline_bold_to_html(safe)

        story.append(Paragraph(safe, body))

    doc.build(story)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main():
    st.set_page_config(page_title="CheckPoint-AI Evaluator", layout="wide")
    st.title("🖥️ Checkpoint AI: 프로젝트 피드백 자동화를 위한 AI 에이전트")

    st.markdown(
        "Hi, BITAmin🍊!<br>PDF를 업로드하시면 AI Agent 파이프라인 실행 → 평가 → 리포트(마크다운/JSON/PDF)를 제공합니다.<br>피드백을 기반으로 프로젝트를 확장시켜 수상을 노려보세요🧡 ",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("중간평가할 PDF를 업로드하세요.", type=["pdf"])

    with st.sidebar:
        st.header("설정")
        output_root = st.text_input("output directory", value="output")
        run_btn = st.button("🚀 평가하기", type="primary", disabled=(uploaded is None))

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if run_btn and uploaded is not None:
        # 1) 임시 파일로 저장
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / uploaded.name
            pdf_path.write_bytes(uploaded.getbuffer())

            st.info(f"업로드 파일 저장 완료: {pdf_path.name}")

            # 2) 파이프라인 실행
            st.write("### 1) 파이프라인 실행")
            prog = st.progress(0)
            status = st.status("파이프라인 실행 중...", expanded=True)

            orchestrator = PipelineOrchestrator()
            prog.progress(10)

            problem_mapping, out_dir, err = orchestrator.process_document(
                str(pdf_path), output_dir=output_root
            )
            prog.progress(60)

            if err:
                st.error(f"❌ 파이프라인 실패 원인: {err}")
                st.stop()

            if not problem_mapping or not out_dir:
                status.update(label="❌ 파이프라인 실패", state="error")
                st.stop()

            doc_id = problem_mapping.document_id
            status.write(f"✅ 파이프라인 완료. document_id={doc_id}")
            prog.progress(70)

            # 3) 평가 실행
            st.write("### 2) 평가 실행")
            evaluator = EvaluationOrchestrator(orchestrator.config)
            eval_report = evaluator.evaluate(
                document_id=doc_id, output_dir=str(out_dir)
            )
            prog.progress(85)

            # 4) 평가 리포트 저장
            # main_pipeline.py에서는 reports_dir를 따로 둠(권장)
            reports_dir = Path(out_dir) / "reports"
            reports_dir.mkdir(exist_ok=True)

            report_gen = ReportGenerator()
            try:
                # 레포에 따라 시그니처가 다를 수 있어 안전하게 처리
                report_gen.save_all_formats(
                    eval_report, reports_dir, Path(uploaded.name).stem
                )
            except TypeError:
                report_gen.save_all_formats(eval_report, reports_dir)

            prog.progress(95)
            status.update(label="✅ 평가 완료", state="complete")
            prog.progress(100)

            # 5) 마크다운 -> PDF 변환 추가
            # 저장된 markdown 파일들을 읽어서 PDF로 변환
            feedback_md = (
                reports_dir
                / f"{Path(uploaded.name).stem}_{doc_id}_evaluation_feedback.md"
            )
            checklist_md = (
                reports_dir
                / f"{Path(uploaded.name).stem}_{doc_id}_evaluation_checklist.md"
            )

            feedback_pdf = (
                reports_dir
                / f"{Path(uploaded.name).stem}_{doc_id}_evaluation_feedback.pdf"
            )
            checklist_pdf = (
                reports_dir
                / f"{Path(uploaded.name).stem}_{doc_id}_evaluation_checklist.pdf"
            )

            if feedback_md.exists():
                md_to_pdf(
                    read_text(feedback_md),
                    str(feedback_pdf),
                    title="Evaluation Feedback",
                )
            if checklist_md.exists():
                md_to_pdf(
                    read_text(checklist_md),
                    str(checklist_pdf),
                    title="Evaluation Checklist",
                )

            st.session_state.last_result = {
                "doc_id": doc_id,
                "out_dir": str(out_dir),
                "reports_dir": str(reports_dir),
                "pdf_name": Path(uploaded.name).stem,
            }

    # 결과 뷰어
    result = st.session_state.last_result
    if result:
        doc_id = result["doc_id"]
        reports_dir = Path(result["reports_dir"])
        pdf_name = result["pdf_name"]

        st.divider()
        st.subheader("📊 결과 보기")

        # 파일 경로들
        report_json = reports_dir / f"{pdf_name}_{doc_id}_evaluation_report.json"
        checklist_md = reports_dir / f"{pdf_name}_{doc_id}_evaluation_checklist.md"
        feedback_md = reports_dir / f"{pdf_name}_{doc_id}_evaluation_feedback.md"
        checklist_pdf = reports_dir / f"{pdf_name}_{doc_id}_evaluation_checklist.pdf"
        feedback_pdf = reports_dir / f"{pdf_name}_{doc_id}_evaluation_feedback.pdf"

        # 요약
        if report_json.exists():
            report = json.loads(report_json.read_text(encoding="utf-8"))
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Score", f"{report.get('total_score', 0):.1f}/100")
            col2.metric("Pass Rate", f"{report.get('overall_pass_rate', 0)*100:.1f}%")
            passed_categories = len(report.get("categories", []))
            col3.metric("Categories", f"{passed_categories}/7")

        # ✅ JSON 탭 제거 (요청 반영)
        tabs = st.tabs(["피드백(마크다운)", "체크리스트(마크다운)", "다운로드"])

        with tabs[0]:
            st.markdown(read_text(feedback_md) or "_feedback.md가 없습니다._")

        with tabs[1]:
            st.markdown(read_text(checklist_md) or "_checklist.md가 없습니다._")

        with tabs[2]:
            # 다운로드 버튼들
            if feedback_md.exists():
                st.download_button(
                    "⬇️ feedback.md",
                    data=feedback_md.read_bytes(),
                    file_name=feedback_md.name,
                )
            if checklist_md.exists():
                st.download_button(
                    "⬇️ checklist.md",
                    data=checklist_md.read_bytes(),
                    file_name=checklist_md.name,
                )
            if report_json.exists():
                st.download_button(
                    "⬇️ report.json",
                    data=report_json.read_bytes(),
                    file_name=report_json.name,
                )

            if feedback_pdf.exists():
                st.download_button(
                    "⬇️ feedback.pdf",
                    data=feedback_pdf.read_bytes(),
                    file_name=feedback_pdf.name,
                )
            if checklist_pdf.exists():
                st.download_button(
                    "⬇️ checklist.pdf",
                    data=checklist_pdf.read_bytes(),
                    file_name=checklist_pdf.name,
                )


if __name__ == "__main__":
    main()
