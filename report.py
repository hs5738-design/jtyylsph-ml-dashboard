# report.py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table

def generate_pdf_report(drift, fairness, stability, filename="risk_report.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    # =============================
    # TITLE
    # =============================
    content.append(Paragraph("AI Governance Risk Report", styles["Title"]))
    content.append(Spacer(1, 12))

    # =============================
    # SUMMARY SECTION
    # =============================
    content.append(Paragraph("Executive Summary", styles["Heading2"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph(
        "This report evaluates model performance across drift, fairness, "
        "and system stability dimensions.", styles["Normal"]
    ))
    content.append(Spacer(1, 12))

    # =============================
    # METRICS
    # =============================
    content.append(Paragraph("Key Metrics", styles["Heading2"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph(f"Drift Score: {round(drift,3)}", styles["Normal"]))
    content.append(Paragraph(f"Fairness Gap: {round(fairness,3)}", styles["Normal"]))
    content.append(Paragraph(f"System Stability: {round(stability,3)}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # =============================
    # RISK ASSESSMENT
    # =============================
    content.append(Paragraph("Risk Assessment", styles["Heading2"]))
    content.append(Spacer(1, 8))

    if drift > 0.3:
        content.append(Paragraph("⚠ Data drift detected. Model retraining recommended.", styles["Normal"]))
    else:
        content.append(Paragraph("✔ Drift within acceptable range.", styles["Normal"]))

    if fairness > 0.1:
        content.append(Paragraph("⚠ Potential bias detected. Fairness audit required.", styles["Normal"]))
    else:
        content.append(Paragraph("✔ Fairness within acceptable range.", styles["Normal"]))

    if stability < 0.5:
        content.append(Paragraph("❌ System instability detected. Immediate action required.", styles["Normal"]))
    else:
        content.append(Paragraph("✔ System stable.", styles["Normal"]))

    content.append(Spacer(1, 12))

    # =============================
    # FINAL VERDICT
    # =============================
    content.append(Paragraph("Final Verdict", styles["Heading2"]))
    content.append(Spacer(1, 8))

    if stability < 0.5:
        verdict = "HIGH RISK — Deployment not recommended."
    elif drift > 0.3 or fairness > 0.1:
        verdict = "MEDIUM RISK — Monitoring required."
    else:
        verdict = "LOW RISK — System acceptable."

    content.append(Paragraph(verdict, styles["Normal"]))

    # =============================
    # BUILD PDF
    # =============================
    doc.build(content)

    return filename
