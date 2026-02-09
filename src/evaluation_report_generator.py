"""
Evaluation Report Generator
Generates evaluation reports in various formats
"""
import json
from pathlib import Path
from typing import Optional

from src.models import EvaluationReport
from src.utils import setup_logger

logger = setup_logger(__name__)


class ReportGenerator:
    """Generate evaluation reports"""

    def save_all_formats(
        self,
        report: EvaluationReport,
        output_dir: Optional[Path],
        pdf_name: Optional[str] = None
    ):
        """
        Save report in all formats

        Args:
            report: EvaluationReport to save
            output_dir: Output directory
            pdf_name: Original PDF name (optional)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate base filename
        if pdf_name:
            base_name = f"{pdf_name}_{report.document_id}"
        else:
            base_name = report.document_id

        # 1. Save JSON
        json_path = output_path / f"{base_name}_evaluation_report.json"
        self._save_json(report, json_path)
        logger.info(f"Saved JSON report: {json_path}")

        # 2. Save Markdown checklist
        md_path = output_path / f"{base_name}_evaluation_checklist.md"
        self._save_checklist_markdown(report, md_path)
        logger.info(f"Saved Markdown checklist: {md_path}")

        # 3. Save detailed feedback
        feedback_path = output_path / f"{base_name}_evaluation_feedback.md"
        self._save_detailed_feedback(report, feedback_path)
        logger.info(f"Saved detailed feedback: {feedback_path}")

    def _save_json(self, report: EvaluationReport, path: Path):
        """Save report as JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

    def _save_checklist_markdown(self, report: EvaluationReport, path: Path):
        """Save checklist as Markdown"""
        md = self.generate_checklist_markdown(report)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)

    def _save_detailed_feedback(self, report: EvaluationReport, path: Path):
        """Save detailed feedback as Markdown"""
        md = self.generate_detailed_feedback(report)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)

    def generate_checklist_markdown(self, report: EvaluationReport) -> str:
        """
        Generate checklist markdown

        Returns:
            Markdown string with checklist format
        """
        md = f"""# 평가 체크리스트 (Evaluation Checklist)

**문서 ID**: `{report.document_id}`  
**평가 일시**: {report.timestamp}  
**총점**: **{report.total_score:.1f}** / 100  
**전체 통과율**: {report.overall_pass_rate * 100:.1f}%

---

"""

        for category in report.categories:
            # Category header
            md += f"\n## {category.category}"

            if category.weight > 0:
                md += f" (가중치: {category.weight * 100:.0f}%, 점수: {category.score:.1f}점)"

            md += f"\n\n**통과**: {category.pass_count}/{category.total_count} " \
                  f"({category.pass_rate * 100:.1f}%)\n\n"

            # Checklist items
            for item in category.checklist_items:
                status = "✅" if item.result else "❌"
                confidence_emoji = "🟢" if item.confidence >= 0.7 else "🟡" if item.confidence >= 0.4 else "🔴"

                md += f"{status} **{item.subcategory}**: {item.question}\n"
                md += f"   - **판단** ({confidence_emoji} {item.confidence:.2f}): {item.reasoning}\n"

                if item.evidence:
                    md += f"   - **근거**:\n"
                    for evidence in item.evidence[:3]:  # Limit to 3
                        truncated = evidence[:150] + "..." if len(evidence) > 150 else evidence
                        md += f"     - {truncated}\n"

                if item.search_results:
                    md += f"   - **검색 결과** ({len(item.search_results)}개):\n"
                    for result in item.search_results[:2]:
                        title = result.get('title', 'No title')[:80]
                        url = result.get('url', '')
                        md += f"     - [{title}]({url})\n"

                md += "\n"

        # Summary
        md += "\n---\n\n## 📊 요약\n\n"
        md += f"- **평가 항목 수**: {report.summary['total_items']}개\n"
        md += f"- **통과 항목 수**: {report.summary['total_passed']}개\n"
        md += f"- **카테고리별 점수**:\n"

        for cat_name, cat_score in report.summary['category_scores'].items():
            md += f"  - {cat_name}: {cat_score:.1f}점\n"

        return md

    def generate_detailed_feedback(self, report: EvaluationReport) -> str:
        """
        Generate detailed feedback report

        Returns:
            Markdown string with detailed feedback
        """
        md = f"""# 상세 평가 피드백 (Detailed Evaluation Feedback)

**문서 ID**: `{report.document_id}`  
**평가 일시**: {report.timestamp}  
**총점**: **{report.total_score:.1f}** / 100

---

## 🎯 종합 평가

총 {report.summary['total_items']}개 항목 중 {report.summary['total_passed']}개를 통과했습니다 \
({report.overall_pass_rate * 100:.1f}%).

"""

        # Strengths
        md += "\n### ✅ 강점\n\n"
        passed_items = []
        for category in report.categories:
            for item in category.checklist_items:
                if item.result:
                    passed_items.append((category.category, item))

        if passed_items:
            for cat_name, item in passed_items[:5]:  # Top 5
                md += f"- **[{cat_name}] {item.subcategory}**: {item.reasoning}\n"
        else:
            md += "- 통과한 항목이 없습니다.\n"

        # Weaknesses
        md += "\n### ❌ 개선 필요 사항\n\n"
        if report.recommendations:
            for rec in report.recommendations[:10]:  # Top 10
                md += f"- {rec}\n"
        else:
            md += "- 모든 항목을 통과했습니다!\n"

        # Category breakdown
        md += "\n---\n\n## 📋 카테고리별 상세 분석\n\n"

        for category in report.categories:
            md += f"\n### {category.category}\n\n"

            if category.weight > 0:
                md += f"**가중치**: {category.weight * 100:.0f}%  \n"
                md += f"**획득 점수**: {category.score:.1f} / {category.weight * 100:.1f}  \n"

            md += f"**통과율**: {category.pass_rate * 100:.1f}% ({category.pass_count}/{category.total_count})\n\n"

            # Failed items first
            failed = [item for item in category.checklist_items if not item.result]
            if failed:
                md += "**❌ 미통과 항목**:\n\n"
                for item in failed:
                    md += f"- **{item.subcategory}**: {item.question}\n"
                    md += f"  - {item.reasoning}\n"
                    if item.evidence:
                        md += f"  - 근거: {item.evidence[0][:100]}...\n"
                    md += "\n"

            # Passed items
            passed = [item for item in category.checklist_items if item.result]
            if passed:
                md += "**✅ 통과 항목**:\n\n"
                for item in passed:
                    md += f"- **{item.subcategory}**: {item.reasoning}\n"

        # Recommendations
        md += "\n---\n\n## 💡 개선 권장사항\n\n"

        if report.recommendations:
            priority_recs = self._prioritize_recommendations(report)

            md += "### 우선순위 높음\n\n"
            for rec in priority_recs['high']:
                md += f"1. {rec}\n"

            if priority_recs['medium']:
                md += "\n### 우선순위 중간\n\n"
                for rec in priority_recs['medium']:
                    md += f"- {rec}\n"
        else:
            md += "모든 평가 항목을 통과했습니다! 🎉\n"

        return md

    def _prioritize_recommendations(self, report: EvaluationReport) -> dict:
        """Prioritize recommendations by category weight"""
        high = []
        medium = []

        for rec in report.recommendations:
            # Check if from high-weight category
            if any(cat in rec for cat in ['주제 선정', '알고리즘 설계', '문제 해결력']):
                high.append(rec)
            else:
                medium.append(rec)

        return {'high': high[:5], 'medium': medium[:5]}
