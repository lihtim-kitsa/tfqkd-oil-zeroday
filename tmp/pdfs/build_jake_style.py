from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

out=Path('output/pdf/Mithil_Astik_CERN_Internship_Resume.pdf')
fontdir=Path(r'C:\Windows\Fonts')
pdfmetrics.registerFont(TTFont('Arial',str(fontdir/'arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold',str(fontdir/'arialbd.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Italic',str(fontdir/'ariali.ttf')))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
black=colors.black; gray=colors.HexColor('#222222')
W,H=letter; left=38; right=38; width=W-left-right; y=H-27
c=canvas.Canvas(str(out),pagesize=letter,pageCompression=1)
c.setTitle('Mithil Hardik Astik - CERN Internship Resume')
c.setAuthor('Mithil Hardik Astik')
base=dict(fontName='Arial',fontSize=9.3,leading=10.65,textColor=gray,spaceAfter=0)
st=ParagraphStyle('base',**base)
bold=ParagraphStyle('bold',fontName='Arial-Bold',fontSize=9.35,leading=10.3,textColor=black)
ital=ParagraphStyle('ital',fontName='Arial-Italic',fontSize=8.7,leading=9.4,textColor=gray)
rightstyle=ParagraphStyle('right',parent=bold,alignment=TA_RIGHT)
rightital=ParagraphStyle('rightital',parent=ital,alignment=TA_RIGHT)
bullet=ParagraphStyle('bullet',fontName='Arial',fontSize=9.15,leading=10.35,textColor=gray,leftIndent=12,firstLineIndent=-8,spaceAfter=0.8)
section_style=ParagraphStyle('section',fontName='Arial-Bold',fontSize=11.2,leading=12.2,textColor=black)

def P(text,style=st): return Paragraph(text,style)
def draw_para(text,style=st,x=left,w=width,gap=0):
    global y
    p=P(text,style); _,h=p.wrap(w,1000); y-=h; p.drawOn(c,x,y); y-=gap
    if y < 18: raise RuntimeError(f'Content overflow: y={y}')

def draw_section(label):
    global y
    y-=3.2
    p=P(label.upper(),section_style); _,h=p.wrap(width,100); y-=h; p.drawOn(c,left,y)
    c.setLineWidth(.45); c.setStrokeColor(black); c.line(left,y-2,width+left,y-2); y-=6.5

def draw_entry(title,right1,sub='',right2=''):
    global y
    data=[[P(title,bold),P(right1,rightstyle)]]
    if sub or right2: data.append([P(sub,ital),P(right2,rightital)])
    t=Table(data,colWidths=[width*.75,width*.25],hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
    _,h=t.wrap(width,1000); y-=h; t.drawOn(c,left,y); y-=1.7

def draw_bullet(txt): draw_para(txt,bullet,gap=.25)

# Centered Jake-style name and contact row
c.setFont('Arial-Bold',21); c.drawCentredString(W/2,H-30,'Mithil Hardik Astik')
contact=ParagraphStyle('contact',fontName='Arial',fontSize=9.1,leading=10.5,alignment=TA_CENTER,textColor=gray)
draw_para('+91 9154312220 $|$ <link href="mailto:astik.mithil@gmail.com">astik.mithil@gmail.com</link> $|$ <link href="https://linkedin.com/in/mithil-astik">linkedin.com/in/mithil-astik</link> $|$ <link href="https://github.com/lihtim-kitsa">github.com/lihtim-kitsa</link> $|$ <link href="https://mithil-astik.vercel.app">Portfolio</link>',contact,x=left,w=width,gap=1.5)

draw_section('Education')
draw_entry('BITS Pilani, Hyderabad Campus','Hyderabad, India','Integrated M.Sc. Physics and B.E. Electrical &amp; Electronics Engineering; CGPA: 6.98','Aug. 2024 -- Present')
draw_bullet('Relevant study: Quantum Information and Computing, Computational Physics, Quantum Mechanics, Electromagnetism and Optics.')

draw_section('Research')
draw_entry('LHC Olympics 2020 R&amp;D Anomaly-Detection Benchmark','2026','Sole-authored manuscript; simulated collider events','Python, PyTorch, scikit-learn')
draw_bullet('Benchmarked 13 unsupervised, semi-supervised and supervised detectors; evaluated contamination, label budget, dijet-mass sculpting and transfer to a held-out three-prong signal topology.')
draw_bullet('Deep SAD reached AUC 0.934 ± 0.001 with 100 labelled anomalies and 0.5% training contamination; held-out-topology AUC was 0.824 ± 0.008. Injected-signal benchmark exceeded 8σ at 1% background efficiency. Simulation study.')

draw_entry('Simulation-Based Detection of Optical Injection-Locking Anomalies in TF-QKD','2026','M. H. Astik and T. S. L. Radhika; manuscript','Python, Lang--Kobayashi dynamics')
draw_bullet('Built a stochastic laser and decoy-state twin-field QKD simulator; generated 22 million pulses with randomized FIM, a TWIRL frequency-sweep proxy and 10 held-out synthetic attack mechanisms. Compared semi-supervised Deep SVDD-type and XGBoost detectors.')
draw_bullet('At 100 km, pooled zero-day AUROC was 0.925 (Deep SVDD-type) and 0.935 (XGBoost); identified carrier-density modulation as a blind spot (AUROC 0.334/0.299) and non-monotonic transfer across laser settings. No hardware validation.')

draw_entry('ML-Based Attack Detection for Twin-Field QKD with Optical Injection Locking','2026','M. H. Astik and T. S. L. Radhika; ICICSCS 2026','Python, machine learning')
draw_bullet('Compared classical and quantum classifiers on 40,000 simulated telemetry samples; XGBoost reached 87.52% test accuracy, while drift augmentation reduced simulated false positives from 99.88% to 2.48%.')

draw_section('Projects')
draw_entry('APEX-PINN: Physics-Informed Neural PDE Solvers  |  Python, PyTorch','In progress')
draw_bullet('Developing coupled fluid, heat-transfer and magnetic-induction PDE solvers; implemented divergence-free parameterizations, adaptive collocation, multi-fidelity training and inverse parameter estimation.')
draw_entry('QKD-SecLab and PIC QKD Simulation Reproduction  |  Python, PyTorch','Course / ongoing')
draw_bullet('Built a modular attack-and-detector lab across BB84, MDI-QKD and TF-QKD; reproduced published PIC-based QKD simulation curves under detector-blinding, Trojan-horse and backflash attacks.')

draw_section('Experience & Leadership')
draw_entry('Quantum Computing Team Lead, IEEE Student Branch, BPHC','Feb. 2025 -- Present','BITS Pilani, Hyderabad Campus','India')
draw_bullet('Led Qiskit activities; secured an IBM Quantum-sponsored partnership and led Quantumverse Vol. 3 with ATMoS ’25.')
draw_entry('Summer Intern','May 2026 -- Jul. 2026','Beijan, BITS Pilani, Hyderabad','India')
draw_bullet('Developed a React hardware-telemetry viewer and configured edge-computing interfaces for GPS-denied navigation.')
draw_entry('Freelance Full-Stack Developer','Feb. 2026 -- Present','Multiple clients','Remote')
draw_bullet('Built a retail inventory and ordering application using Next.js, PostgreSQL and TypeScript.')

draw_section('Technical Skills')
draw_para('<b>Scientific computing:</b> Python, C/C++, NumPy, PyTorch, numerical simulation, PDE modelling, ANSYS Lumerical, FastJet<br/><b>Machine learning:</b> scikit-learn, XGBoost, LightGBM, SHAP, anomaly detection, MLflow<br/><b>Quantum:</b> Qiskit, PennyLane, quantum-kernel methods &nbsp;&nbsp; <b>Software:</b> Git, TypeScript, JavaScript, React, Node.js, FastAPI, PostgreSQL, SQLite')
c.save()
print(out.resolve(), 'remaining vertical space',round(y,1),'pt')
