from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

def generate_pdf_report(drift, fairness, stability, filename="risk_report.pdf"):

    file_path = os.path.join("/tmp", filename)  # ✅ FIXED

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI Governance Risk Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Executive Summary", styles["Heading2"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph(
        "This report evaluates model performance across drift, fairness, and system stability.",
        styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Key Metrics", styles["Heading2"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph(f"Drift Score: {round(drift,3)}", styles["Normal"]))
    content.append(Paragraph(f"Fairness Gap: {round(fairness,3)}", styles["Normal"]))
    content.append(Paragraph(f"System Stability: {round(stability,3)}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Risk Assessment", styles["Heading2"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph(
        "⚠ Data drift detected." if drift > 0.3 else "✔ Drift acceptable.",
        styles["Normal"]
    ))

    content.append(Paragraph(
        "⚠ Bias risk detected." if fairness > 0.1 else "✔ Fairness acceptable.",
        styles["Normal"]
    ))

    content.append(Paragraph(
        "❌ System unstable." if stability < 0.5 else "✔ System stable.",
        styles["Normal"]
    ))

    content.append(Spacer(1, 12))

    verdict = (
        "HIGH RISK — Deployment not recommended."
        if stability < 0.5 else
        "MEDIUM RISK — Monitoring required."
        if drift > 0.3 or fairness > 0.1 else
        "LOW RISK — System acceptable."
    )

    content.append(Paragraph("Final Verdict", styles["Heading2"]))
    content.append(Spacer(1, 8))
    content.append(Paragraph(verdict, styles["Normal"]))

    doc.build(content)

    return file_path   # ✅ IMPORTANT
