# report.py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def generate_pdf_report(drift, fairness, stability, filename="risk_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI Risk Monitoring Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Drift Score: {round(drift,3)}", styles["Normal"]))
    content.append(Paragraph(f"Fairness Gap: {round(fairness,3)}", styles["Normal"]))
    content.append(Paragraph(f"System Stability: {round(stability,3)}", styles["Normal"]))

    content.append(Spacer(1, 12))

    # Interpretation
    if drift > 0.3:
        content.append(Paragraph("⚠ Drift detected — retraining recommended.", styles["Normal"]))

    if fairness > 0.1:
        content.append(Paragraph("⚠ Bias risk detected — audit model.", styles["Normal"]))

    if stability < 0.5:
        content.append(Paragraph("❌ System unstable — immediate action required.", styles["Normal"]))
    else:
        content.append(Paragraph("✅ System stable.", styles["Normal"]))

    doc.build(content)

    return filename
