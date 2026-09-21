"""PDF report generation (Module 13 export) with reportlab."""

from __future__ import annotations

import io
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sentinai.schemas import SEVERITY_NAMES, SeverityLevel
from sentinai.storage.models import AuthorRow, ClassificationRow, PostRow, TargetHitRow


def build_pdf_report(session: Session, days: int = 90) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    since = datetime.utcnow() - timedelta(days=days)
    total = session.scalar(select(func.count()).select_from(ClassificationRow).join(PostRow, PostRow.id == ClassificationRow.post_id).where(PostRow.created_at >= since)) or 0
    toxic = session.scalar(select(func.count()).select_from(ClassificationRow).join(PostRow, PostRow.id == ClassificationRow.post_id).where(PostRow.created_at >= since, ClassificationRow.final_toxicity >= 0.5)) or 0
    by_label = session.execute(select(ClassificationRow.toxicity_label, func.count()).join(PostRow, PostRow.id == ClassificationRow.post_id).where(PostRow.created_at >= since).group_by(ClassificationRow.toxicity_label)).all()
    by_sev = session.execute(select(ClassificationRow.severity_level, func.count()).join(PostRow, PostRow.id == ClassificationRow.post_id).where(PostRow.created_at >= since).group_by(ClassificationRow.severity_level).order_by(ClassificationRow.severity_level)).all()
    top_targets = session.execute(select(TargetHitRow.category, TargetHitRow.label, func.count(), func.avg(TargetHitRow.severity_level)).join(PostRow, PostRow.id == TargetHitRow.post_id).where(PostRow.created_at >= since, TargetHitRow.final_toxicity >= 0.5).group_by(TargetHitRow.category, TargetHitRow.label).order_by(func.count().desc()).limit(12)).all()
    bots = session.scalar(select(func.count()).select_from(AuthorRow).where(AuthorRow.bot_probability >= 0.8)) or 0
    authors = session.scalar(select(func.count()).select_from(AuthorRow)) or 0

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm, title="SentinAI report")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SentinAI — Bias &amp; Hate Speech Intelligence Report", styles["Title"]),
        Paragraph(f"Window: last {days} days · generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC", styles["Normal"]),
        Spacer(1, 0.4 * cm),
        Paragraph("<b>Disclaimer.</b> All figures are model-estimated probabilities produced by automated classifiers. They are not definitive human judgments and must not be used as the sole basis for enforcement decisions.", styles["Italic"]),
        Spacer(1, 0.6 * cm),
        Paragraph("1. Overview", styles["Heading2"]),
    ]
    overview = [["Metric", "Value"], ["Posts analysed", f"{total:,}"], ["Toxic posts (≥ 0.5)", f"{toxic:,} ({(toxic / total * 100) if total else 0:.1f}%)"], ["Distinct authors", f"{authors:,}"], ["Likely bot accounts (P ≥ 0.8)", f"{bots:,}"]]
    story.append(_table(overview, Table, TableStyle, colors))
    story += [Spacer(1, 0.5 * cm), Paragraph("2. Toxicity labels", styles["Heading2"]), _table([["Label", "Posts"]] + [[lab, f"{n:,}"] for lab, n in by_label], Table, TableStyle, colors)]
    story += [Spacer(1, 0.5 * cm), Paragraph("3. Severity distribution", styles["Heading2"]), _table([["Level", "Category", "Posts"]] + [[str(lvl), SEVERITY_NAMES[SeverityLevel(int(lvl))], f"{n:,}"] for lvl, n in by_sev], Table, TableStyle, colors)]
    story += [Spacer(1, 0.5 * cm), Paragraph("4. Most targeted demographics", styles["Heading2"]), _table([["Category", "Target", "Toxic posts", "Avg severity"]] + [[c, l, f"{n:,}", f"{float(s):.2f}"] for c, l, n, s in top_targets], Table, TableStyle, colors)]
    story += [Spacer(1, 0.5 * cm), Paragraph("5. Methodology", styles["Heading2"]), Paragraph("Posts are normalised (URL removal, mention anonymisation, hashtag splitting, emoji description, obfuscation repair), language-identified and de-duplicated (MinHash/LSH). Three parallel tasks then run: toxicity classification, target demographic identification and ordinal severity scoring. Counter-speech and reply-chain context discount quoted or condemned material; multimodal posts merge OCR / vision signals. Posts with confidence in the 0.4–0.6 band are routed to human review.", styles["Normal"])]
    doc.build(story)
    return buf.getvalue()


def _table(data, Table, TableStyle, colors):
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    return t
