import os
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


MODEL_VERSION = "TrustVision DeepfakeResNet v1.0"
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"


def infer_media_type(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "Image"
    if suffix in {".mp4", ".mov", ".avi"}:
        return "Video"
    return "Unknown"


def build_explanation_summary(scan_record) -> str:
    authenticity = float(scan_record.authenticity_score or 0.0)
    confidence = float(scan_record.confidence or 0.0)
    prediction = (scan_record.prediction or "UNKNOWN").upper()
    risk = (scan_record.risk_level or "UNKNOWN").upper()

    if prediction == "FAKE":
        signal = "The detector found manipulation indicators that are more consistent with synthetic or altered media."
    else:
        signal = "The detector found stronger evidence of authentic visual structure than of manipulation."

    confidence_note = (
        f"The current verdict confidence is {confidence:.2f}% with an authenticity score of {authenticity:.2f}%."
    )
    risk_note = f"This scan is categorized as {risk} risk under the current calibration policy."

    return f"{signal} {confidence_note} {risk_note}"


def _draw_wrapped_text(pdf: canvas.Canvas, text: str, x: float, y: float, width: float, font_name="Helvetica", font_size=11, leading=15):
    lines = simpleSplit(text, font_name, font_size, width)
    pdf.setFont(font_name, font_size)
    current_y = y
    for line in lines:
        pdf.drawString(x, current_y, line)
        current_y -= leading
    return current_y


def generate_pdf_report(scan_record):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    margin_x = 18 * mm
    top_y = height - 22 * mm
    content_width = width - (2 * margin_x)

    pdf.setTitle(f"TrustVision Report #{scan_record.id}")

    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, height - 40 * mm, width, 40 * mm, stroke=0, fill=1)
    pdf.setFillColor(colors.HexColor("#34d399"))
    pdf.rect(0, height - 6 * mm, width, 6 * mm, stroke=0, fill=1)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(margin_x, top_y, "TrustVision AI")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(margin_x, top_y - 8 * mm, "Media Authenticity PDF Report")

    section_y = height - 56 * mm
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(margin_x, section_y, "Media Metadata")

    pdf.setFont("Helvetica", 11)
    timestamp = scan_record.timestamp.isoformat() if scan_record.timestamp else "Unavailable"
    metadata_rows = [
        ("Scan ID", str(scan_record.id)),
        ("Filename", scan_record.filename or "Unnamed media"),
        ("Media Type", infer_media_type(scan_record.filename)),
        ("Timestamp", timestamp),
        ("Model Version", MODEL_VERSION),
    ]

    row_y = section_y - 10 * mm
    for label, value in metadata_rows:
        pdf.setFillColor(colors.HexColor("#475569"))
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(margin_x, row_y, f"{label}:")
        pdf.setFillColor(colors.HexColor("#111827"))
        pdf.setFont("Helvetica", 10)
        pdf.drawString(margin_x + 34 * mm, row_y, value)
        row_y -= 7 * mm

    section_y = row_y - 3 * mm
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(margin_x, section_y, "Detection Summary")

    summary_rows = [
        ("Prediction", scan_record.prediction or "Unknown"),
        ("Authenticity Score", f"{float(scan_record.authenticity_score or 0.0):.2f}%"),
        ("Confidence", f"{float(scan_record.confidence or 0.0):.2f}%"),
        ("Risk Assessment", scan_record.risk_level or "Unknown"),
    ]

    row_y = section_y - 10 * mm
    for label, value in summary_rows:
        pdf.setFillColor(colors.HexColor("#475569"))
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(margin_x, row_y, f"{label}:")
        pdf.setFillColor(colors.HexColor("#111827"))
        pdf.setFont("Helvetica", 10)
        pdf.drawString(margin_x + 34 * mm, row_y, value)
        row_y -= 7 * mm

    section_y = row_y - 3 * mm
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(margin_x, section_y, "AI Explanation Summary")

    explanation_summary = build_explanation_summary(scan_record)
    pdf.setFillColor(colors.HexColor("#111827"))
    body_y = section_y - 10 * mm
    final_y = _draw_wrapped_text(
        pdf,
        explanation_summary,
        margin_x,
        body_y,
        content_width,
        font_name="Helvetica",
        font_size=11,
        leading=15,
    )

    footer_y = max(18 * mm, final_y - 18 * mm)
    pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
    pdf.line(margin_x, footer_y + 7 * mm, width - margin_x, footer_y + 7 * mm)
    pdf.setFillColor(colors.HexColor("#64748b"))
    pdf.setFont("Helvetica", 9)
    pdf.drawString(margin_x, footer_y, "Generated by TrustVision AI")
    pdf.drawRightString(width - margin_x, footer_y, "For verification support, retain the scan ID in this document.")

    pdf.save()
    pdf_bytes = pdf_buffer.getvalue()

    report_filename = f"trustvision_report_{scan_record.id}.pdf"
    report_path = REPORTS_DIR / report_filename
    with report_path.open("wb") as handle:
        handle.write(pdf_bytes)

    return pdf_bytes, report_filename, explanation_summary
