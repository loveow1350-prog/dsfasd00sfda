import os
import json
import tempfile
from pathlib import Path
import sys
import re
import asyncio
import smtplib
from email.message import EmailMessage

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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# Windows에서 Streamlit/asyncio 이슈 방지용 (이미 적용하셨던 부분 유지)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ---------------------------
# 폰트 설정 (여기만 본인 환경에 맞게)
# 프로젝트 폴더 내 fonts 폴더에 ttf 넣고 쓰는 방식 권장
# 예: CheckPoint-AI/fonts/NotoSansKR-VariableFont_wght.ttf
# ---------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_FONT_PATH = APP_DIR / "fonts" / "NotoSansKR-VariableFont_wght.ttf"
FONT_NAME_REGULAR = "NotoSansKR"


def register_korean_font(font_path: Path = DEFAULT_FONT_PATH) -> bool:
    """ReportLab에서 한글 폰트 등록. 성공하면 True."""
    try:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(FONT_NAME_REGULAR, str(font_path)))
            return True
        return False
    except Exception:
        return False


# ---------------------------
# Markdown -> ReportLab Paragraph 변환 보조
# ---------------------------
def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_bold_to_reportlab_markup(text: str) -> str:
    """
    Markdown **bold** 를 ReportLab Paragraph 마크업(<b>...</b>)으로 변환.
    - escape는 먼저 해주고 변환합니다.
    """
    # **...** -> <b>...</b>
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def md_to_pdf(
    md_text: str,
    out_path: str,
    title: str = "Report",
    font_path: Path = DEFAULT_FONT_PATH,
):
    """
    아주 가벼운 Markdown -> PDF 변환기
    - heading/bullet/codeblock 중심
    - **bold** 처리: PDF에서도 굵게
    - 한글 폰트 등록 시 한글 깨짐 방지
    """
    font_ok = register_korean_font(font_path)
    styles = getSampleStyleSheet()

    base_font = FONT_NAME_REGULAR if font_ok else styles["BodyText"].fontName

    h1 = ParagraphStyle(
        "h1", parent=styles["Heading1"], spaceAfter=10, fontName=base_font
    )
    h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"], spaceAfter=8, fontName=base_font
    )
    h3 = ParagraphStyle(
        "h3", parent=styles["Heading3"], spaceAfter=6, fontName=base_font
    )
    body = ParagraphStyle(
        "body", parent=styles["BodyText"], leading=14, spaceAfter=6, fontName=base_font
    )
    mono = ParagraphStyle(
        "mono", parent=styles["Code"], leading=12, spaceAfter=6, fontName=base_font
    )

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    story = [Paragraph(escape_html(title), h1), Spacer(1, 8)]

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
            story.append(
                Paragraph(
                    md_bold_to_reportlab_markup(escape_html(line[2:].strip())), h1
                )
            )
            story.append(Spacer(1, 6))
            continue
        if line.startswith("## "):
            story.append(
                Paragraph(
                    md_bold_to_reportlab_markup(escape_html(line[3:].strip())), h2
                )
            )
            story.append(Spacer(1, 4))
            continue
        if line.startswith("### "):
            story.append(
                Paragraph(
                    md_bold_to_reportlab_markup(escape_html(line[4:].strip())), h3
                )
            )
            story.append(Spacer(1, 4))
            continue

        # bullet
        if line.strip().startswith("- "):
            bullet_text = line.strip()[2:]
            safe = md_bold_to_reportlab_markup(escape_html(bullet_text))
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
        safe = md_bold_to_reportlab_markup(escape_html(line))
        story.append(Paragraph(safe, body))

    doc.build(story)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ---------------------------
# SMTP 메일 발송
# ---------------------------
def send_email_smtp(
    smtp_host: str,
    smtp_port: int,
    use_tls: bool,
    username: str,
    password: str,
    mail_from: str,
    mail_to: str,
    subject: str,
    body: str,
    attachments: list[Path],
):
    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg["Subject"] = subject
    msg.set_content(body)

    for p in attachments:
        if not p.exists():
            continue
        data = p.read_bytes()
        # 간단히 확장자 기준으로
        if p.suffix.lower() == ".pdf":
            maintype, subtype = "application", "pdf"
        elif p.suffix.lower() == ".json":
            maintype, subtype = "application", "json"
        else:
            maintype, subtype = "text", "plain"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=p.name)

    if use_tls:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            if username:
                s.login(username, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            if username:
                s.login(username, password)
            s.send_message(msg)


def main():
    st.set_page_config(page_title="CheckPoint-AI Evaluator", layout="wide")
    st.title("🖥️ Checkpoint AI: 프로젝트 피드백 자동화를 위한 AI 에이전트")

    # "Hi, BITAmin!" 이후 한 줄 띄우기: markdown + <br> 사용
    st.markdown(
        "Hi, BITAmin🍊!<br><br>"
        "PDF를 업로드하시면 AI Agent 파이프라인 실행 → 평가 → 리포트(마크다운/JSON/PDF)를 제공합니다.<br>"
        "피드백을 기반으로 프로젝트를 확장시켜 수상을 노려보세요🧡",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("중간평가할 PDF를 업로드하세요.", type=["pdf"])

    with st.sidebar:
        st.header("설정")
        output_root = st.text_input("output directory", value="output")
        # 폰트 경로를 UI에서 바꿀 수 있게 (원하면 고정해도 됨)
        font_path_str = st.text_input(
            "PDF 한글 폰트(ttf) 경로", value=str(DEFAULT_FONT_PATH)
        )
        run_btn = st.button("🚀 평가하기", type="primary", disabled=(uploaded is None))

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if run_btn and uploaded is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / uploaded.name
            pdf_path.write_bytes(uploaded.getbuffer())
            st.info(f"업로드 파일 저장 완료: {pdf_path.name}")

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

            st.write("### 2) 평가 실행")
            evaluator = EvaluationOrchestrator(orchestrator.config)
            eval_report = evaluator.evaluate(
                document_id=doc_id, output_dir=str(out_dir)
            )
            prog.progress(85)

            reports_dir = Path(out_dir) / "reports"
            reports_dir.mkdir(exist_ok=True)

            report_gen = ReportGenerator()
            try:
                report_gen.save_all_formats(
                    eval_report, reports_dir, Path(uploaded.name).stem
                )
            except TypeError:
                report_gen.save_all_formats(eval_report, reports_dir)

            prog.progress(95)
            status.update(label="✅ 평가 완료", state="complete")
            prog.progress(100)

            # PDF 변환
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

            font_path = Path(font_path_str)

            if feedback_md.exists():
                md_to_pdf(
                    read_text(feedback_md),
                    str(feedback_pdf),
                    title="Evaluation Feedback",
                    font_path=font_path,
                )
            if checklist_md.exists():
                md_to_pdf(
                    read_text(checklist_md),
                    str(checklist_pdf),
                    title="Evaluation Checklist",
                    font_path=font_path,
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
            col3.metric("Categories", f"{passed_categories}/7")  # ✅ 5/7 형태

        # ✅ JSON 탭 제거 반영
        tabs = st.tabs(
            [
                "피드백(마크다운)",
                "체크리스트(마크다운)",
                "다운로드",
                "메일 보내기(SMTP)",
            ]
        )

        with tabs[0]:
            st.markdown(read_text(feedback_md) or "_feedback.md가 없습니다._")

        with tabs[1]:
            st.markdown(read_text(checklist_md) or "_checklist.md가 없습니다._")

        with tabs[2]:
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

        with tabs[3]:
            st.write("로컬에서 SMTP로 파일을 선택해 메일로 보낼 수 있습니다.")

            # 첨부 파일 선택
            candidates = [
                ("feedback.md", feedback_md),
                ("checklist.md", checklist_md),
                ("report.json", report_json),
                ("feedback.pdf", feedback_pdf),
                ("checklist.pdf", checklist_pdf),
            ]
            st.markdown("#### 1) 첨부할 파일 선택")
            selected = []
            cols = st.columns(2)
            for i, (label, path) in enumerate(candidates):
                with cols[i % 2]:
                    if path.exists():
                        if st.checkbox(f"{label} 첨부", value=(label.endswith(".pdf"))):
                            selected.append(path)
                    else:
                        st.caption(f"{label}: 없음")

            st.markdown("#### 2) SMTP 설정")
            smtp_host = st.text_input("SMTP Host", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587, step=1)
            use_tls = st.checkbox("STARTTLS 사용", value=True)

            username = st.text_input("SMTP Username (선택)", value="")
            password = st.text_input(
                "SMTP Password/App Password", value="", type="password"
            )

            st.markdown("#### 3) 메일 내용")
            mail_from = st.text_input("From", value=(username if username else ""))
            mail_to = st.text_input("To", value="")
            subject = st.text_input(
                "Subject", value=f"[CheckPoint-AI] 결과 리포트 ({pdf_name})"
            )
            body = st.text_area(
                "Body",
                value="안녕하세요.\n첨부파일로 평가 결과 리포트를 전달드립니다.\n확인 부탁드립니다.\n감사합니다.",
                height=140,
            )

            if st.button("📧 메일 보내기", type="primary"):
                if not smtp_host or not smtp_port or not mail_to or not mail_from:
                    st.error("SMTP Host/Port, From, To는 필수입니다.")
                elif use_tls and smtp_port == 465:
                    st.warning(
                        "465는 보통 SSL 포트입니다. STARTTLS면 587을 권장합니다."
                    )
                else:
                    try:
                        send_email_smtp(
                            smtp_host=smtp_host,
                            smtp_port=int(smtp_port),
                            use_tls=use_tls,
                            username=username,
                            password=password,
                            mail_from=mail_from,
                            mail_to=mail_to,
                            subject=subject,
                            body=body,
                            attachments=selected,
                        )
                        st.success(f"메일 발송 완료! (첨부 {len(selected)}개)")
                    except Exception as e:
                        st.error(f"메일 발송 실패: {e}")


if __name__ == "__main__":
    main()
