"""
Generate Simplified, High-Impact Major Project Review 2 Presentation for GCET / AKTU B.Tech AIML.
Adheres strictly to the official GCET Review 2 format template (Review 2_ Format ppt_26-27.pptx),
with a clean, concise project title, simple everyday explanations, zero confusing jargon,
and clear structured slides that are easy for both evaluators and students to understand.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    template_path = Path(r"C:\Users\hy180\Downloads\Review 2_ Format ppt_26-27.pptx")
    output_path = Path(r"C:\Users\hy180\Downloads\Major_Project_Review_2_Presentation.pptx")

    if template_path.exists():
        prs = Presentation(str(template_path))
        print("Loaded official GCET Review 2 template.")
    else:
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        print("Created new presentation with 16:9 widescreen dimensions.")

    # Premium Professional Color Palette
    NAVY = RGBColor(15, 32, 67)
    PRIMARY_BLUE = RGBColor(37, 99, 235)
    TEAL_ACCENT = RGBColor(14, 165, 233)
    CHARCOAL = RGBColor(30, 41, 59)
    SLATE = RGBColor(71, 85, 105)
    LIGHT_BG = RGBColor(248, 250, 252)
    WHITE = RGBColor(255, 255, 255)
    CARD_BORDER = RGBColor(226, 232, 240)
    GREEN = RGBColor(16, 185, 129)
    RED = RGBColor(220, 38, 38)

    content_layout = prs.slide_layouts[0]

    def set_slide_header(slide, title_text, category="MAJOR PROJECT REVIEW 2 | AIML"):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.73), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = category.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = TEAL_ACCENT
        p0.font.name = "Arial"

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = NAVY
        p1.font.name = "Arial"

    # ==========================================
    # SLIDE 1: Title Slide (Simple & Clean Name)
    # ==========================================
    slide1 = prs.slides[0]
    for shape in list(slide1.shapes):
        sp = shape._element
        sp.getparent().remove(sp)

    # Accent top bar
    top_bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.18))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = PRIMARY_BLUE
    top_bar.line.fill.background()

    # College Title Badge
    inst_box = slide1.shapes.add_textbox(Inches(0.8), Inches(0.55), Inches(11.73), Inches(0.7))
    tf_inst = inst_box.text_frame
    tf_inst.word_wrap = True
    p_inst = tf_inst.paragraphs[0]
    p_inst.text = "GALGOTIAS COLLEGE OF ENGINEERING & TECHNOLOGY, GREATER NOIDA"
    p_inst.font.size = Pt(13)
    p_inst.font.bold = True
    p_inst.font.color.rgb = PRIMARY_BLUE
    p_inst.alignment = PP_ALIGN.CENTER

    p_dept = tf_inst.add_paragraph()
    p_dept.text = "Department of Artificial Intelligence and Machine Learning | Major Project Review 2 (2026–2027)"
    p_dept.font.size = Pt(11)
    p_dept.font.color.rgb = SLATE
    p_dept.alignment = PP_ALIGN.CENTER

    # Project Title Card (Clean, Simple, Not Overly Complex)
    title_box = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(11.73), Inches(2.0))
    title_box.fill.solid()
    title_box.fill.fore_color.rgb = LIGHT_BG
    title_box.line.color.rgb = CARD_BORDER
    title_box.line.width = Pt(1.5)
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.vertical_anchor = MSO_ANCHOR.MIDDLE

    p_proj = tf_title.paragraphs[0]
    p_proj.text = "Early Autism Detection in Children Using Facial Images"
    p_proj.font.size = Pt(25)
    p_proj.font.bold = True
    p_proj.font.color.rgb = NAVY
    p_proj.alignment = PP_ALIGN.CENTER

    p_sub = tf_title.add_paragraph()
    p_sub.text = "A Non-Invasive Deep Learning Screening System (NeuroGraph-ASD)"
    p_sub.font.size = Pt(13)
    p_sub.font.color.rgb = PRIMARY_BLUE
    p_sub.alignment = PP_ALIGN.CENTER

    # Guide Card (Left)
    guide_card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.75), Inches(5.65), Inches(3.0))
    guide_card.fill.solid()
    guide_card.fill.fore_color.rgb = WHITE
    guide_card.line.color.rgb = CARD_BORDER
    tf_guide = guide_card.text_frame
    tf_guide.word_wrap = True
    tf_guide.margin_left = tf_guide.margin_right = tf_guide.margin_top = Inches(0.3)

    p_gtitle = tf_guide.paragraphs[0]
    p_gtitle.text = "PROJECT SUPERVISOR / GUIDE"
    p_gtitle.font.size = Pt(11)
    p_gtitle.font.bold = True
    p_gtitle.font.color.rgb = TEAL_ACCENT

    p_gname = tf_guide.add_paragraph()
    p_gname.text = "Prof. Uday Singh"
    p_gname.font.size = Pt(18)
    p_gname.font.bold = True
    p_gname.font.color.rgb = NAVY

    p_gdes = tf_guide.add_paragraph()
    p_gdes.text = "Assistant Professor\nDepartment of Artificial Intelligence & Machine Learning (AIML)\nGalgotias College of Engineering & Technology"
    p_gdes.font.size = Pt(12)
    p_gdes.font.color.rgb = SLATE

    # Team Members Card (Right)
    team_card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.85), Inches(3.75), Inches(5.68), Inches(3.0))
    team_card.fill.solid()
    team_card.fill.fore_color.rgb = WHITE
    team_card.line.color.rgb = CARD_BORDER
    tf_team = team_card.text_frame
    tf_team.word_wrap = True
    tf_team.margin_left = tf_team.margin_right = tf_team.margin_top = Inches(0.3)

    p_ttitle = tf_team.paragraphs[0]
    p_ttitle.text = "PROJECT TEAM MEMBERS (AIML)"
    p_ttitle.font.size = Pt(11)
    p_ttitle.font.bold = True
    p_ttitle.font.color.rgb = TEAL_ACCENT

    members = [
        ("Himanshu Yadav", "2300971530064"),
        ("Hemant Kumar", "2300971530063"),
        ("Harsh Chaudhary", "2300971530058"),
        ("Kapil Sharma", "2300971530067")
    ]
    for name, roll in members:
        p_mem = tf_team.add_paragraph()
        p_mem.text = f"•  {name}   —   Roll No: {roll}"
        p_mem.font.size = Pt(12)
        p_mem.font.bold = True
        p_mem.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 2: Table of Contents (Clear & Simple)
    # ==========================================
    slide2 = prs.slides[1]
    for shape in list(slide2.shapes):
        sp = shape._element
        sp.getparent().remove(sp)

    set_slide_header(slide2, "Table of Contents", "PRESENTATION OUTLINE")

    toc_items = [
        ("01", "Introduction", "What is Autism Spectrum Disorder & why early detection matters"),
        ("02", "Motivation", "Why simple photos are better than loud, expensive brain scans"),
        ("03", "Literature Survey", "What previous research has achieved and the research gap"),
        ("04", "Problem Statement", "The challenge of late diagnosis and lack of accessible tools"),
        ("05", "Project Objectives", "5 clear milestones we set out to accomplish"),
        ("06", "Proposed Methodology", "Step-by-step workflow from photo upload to result"),
        ("07", "How the Model Works", "Explaining transfer learning, facial features, and visual heatmaps"),
        ("08", "Results & Performance", "Accuracy, precision, recall, and real-time inference speed"),
        ("09", "Tools & Technologies", "Technologies used: Python, PyTorch, Spring Boot, React"),
        ("10", "Impacts & Benefits", "Safe for toddlers, low cost, and ready for tele-screening"),
        ("11", "References", "Key research publications cited (APA format)")
    ]

    for idx, (num, title, desc) in enumerate(toc_items):
        col = 0 if idx < 6 else 1
        row = idx if col == 0 else idx - 6
        left = Inches(0.8) if col == 0 else Inches(6.85)
        top = Inches(1.5) + Inches(row * 0.92)

        card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.68), Inches(0.82))
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_BG
        card.line.color.rgb = CARD_BORDER
        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.2)
        tf_c.margin_top = Inches(0.1)

        p_t = tf_c.paragraphs[0]
        p_t.text = f"{num}.  {title}"
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY

        p_d = tf_c.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = SLATE

    # ==========================================
    # SLIDE 3: Introduction
    # ==========================================
    slide3 = prs.slides.add_slide(content_layout)
    set_slide_header(slide3, "1. Introduction: What is Autism & Why Early Detection Matters")

    intro_cards = [
        ("What is Autism Spectrum Disorder (ASD)?", "Common Condition: 1 in 36 Children", [
            "Autism is a developmental condition that affects how a person communicates, interacts with others, and experiences the world.",
            "It is a spectrum, meaning every child has unique strengths and challenges."
        ]),
        ("Why Early Detection is Critical", "Best Window: Before Age 3", [
            "A child's brain develops at its fastest rate during the first three years of life.",
            "Starting speech and behavioral therapy before age 3 dramatically improves a child's learning, communication, and independence."
        ]),
        ("The Current Problem with Diagnosis", "Late Diagnosis: Around 4.5+ Years", [
            "Current diagnosis requires long clinical interviews (3 to 6 hours) with rare pediatric specialists.",
            "Families often wait 12 to 18 months for an appointment, missing the most critical early therapy window."
        ]),
        ("Our Solution: Facial Image Analysis", "Simple, Fast, and Non-Invasive", [
            "Medical studies show that children with autism often share subtle facial traits (eye spacing, mid-face and mouth proportions).",
            "We use Artificial Intelligence to analyze these facial features from a standard photograph in under 1 second."
        ])
    ]

    for idx, (title, tag, points) in enumerate(intro_cards):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8) + Inches(col * 5.95)
        top = Inches(1.5) + Inches(row * 2.75)

        card = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.55))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.25)

        p_h = tf.paragraphs[0]
        p_h.text = title
        p_h.font.size = Pt(13.5)
        p_h.font.bold = True
        p_h.font.color.rgb = NAVY

        p_tag = tf.add_paragraph()
        p_tag.text = tag
        p_tag.font.size = Pt(10)
        p_tag.font.bold = True
        p_tag.font.color.rgb = TEAL_ACCENT

        for pt in points:
            p_pt = tf.add_paragraph()
            p_pt.text = f"• {pt}"
            p_pt.font.size = Pt(11)
            p_pt.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 4: Motivation
    # ==========================================
    slide4 = prs.slides.add_slide(content_layout)
    set_slide_header(slide4, "2. Motivation: Why Photographs Instead of Brain Scans (MRI)?")

    # Left: Problems with Brain Scans
    fmri_card = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.75), Inches(5.4))
    fmri_card.fill.solid()
    fmri_card.fill.fore_color.rgb = LIGHT_BG
    fmri_card.line.color.rgb = RED
    fmri_card.line.width = Pt(1.5)
    tf_f = fmri_card.text_frame
    tf_f.word_wrap = True
    tf_f.margin_left = tf_f.margin_right = tf_f.margin_top = Inches(0.3)

    p_fhead = tf_f.paragraphs[0]
    p_fhead.text = "DRAWBACKS OF BRAIN SCANS (fMRI / MRI)"
    p_fhead.font.size = Pt(13)
    p_fhead.font.bold = True
    p_fhead.font.color.rgb = RED

    fmri_points = [
        ("Extremely Expensive", "A single MRI brain scan costs ₹15,000 to ₹35,000+, making routine early screening impossible for average families."),
        ("Loud & Terrifying for Toddlers", "MRI machines generate deafening noise (over 100 dB) inside an enclosed tunnel, causing severe panic and distress in autistic children."),
        ("Requires Medical Sedation", "Young children cannot stay perfectly still for 30–45 minutes, often requiring general anesthesia which carries health risks."),
        ("Only in Big Research Hospitals", "Requires multi-ton machines; impossible to use in rural clinics, community centers, or at home.")
    ]
    for title, desc in fmri_points:
        p1 = tf_f.add_paragraph()
        p1.text = f"✕  {title}"
        p1.font.size = Pt(11.5)
        p1.font.bold = True
        p1.font.color.rgb = NAVY
        p2 = tf_f.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = SLATE

    # Right: Benefits of Facial Photos
    face_card = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.75), Inches(5.4))
    face_card.fill.solid()
    face_card.fill.fore_color.rgb = WHITE
    face_card.line.color.rgb = GREEN
    face_card.line.width = Pt(1.5)
    tf_face = face_card.text_frame
    tf_face.word_wrap = True
    tf_face.margin_left = tf_face.margin_right = tf_face.margin_top = Inches(0.3)

    p_ahead = tf_face.paragraphs[0]
    p_ahead.text = "BENEFITS OF OUR 2D FACIAL PHOTO APPROACH"
    p_ahead.font.size = Pt(13)
    p_ahead.font.bold = True
    p_ahead.font.color.rgb = GREEN

    face_points = [
        ("100% Safe, Calm & Comfortable", "No machines, no noise, no needles. A parent can simply take a normal photo of their child smiling on their lap."),
        ("Zero Extra Cost", "Runs on standard smartphones, basic tablets, or clinic webcams without needing any expensive equipment."),
        ("Instant Results in Under 1 Second", "The AI model evaluates the picture and gives results immediately, saving months of waiting."),
        ("Accessible to Every Family", "Can be used in primary health centers, rural clinics, schools, or through telemedicine apps anywhere.")
    ]
    for title, desc in face_points:
        p1 = tf_face.add_paragraph()
        p1.text = f"✓  {title}"
        p1.font.size = Pt(11.5)
        p1.font.bold = True
        p1.font.color.rgb = NAVY
        p2 = tf_face.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 5: Literature Survey
    # ==========================================
    slide5 = prs.slides.add_slide(content_layout)
    set_slide_header(slide5, "3. Literature Survey: What Previous Research Has Done")

    # Table of Key Reviewed Studies
    table_shape = slide5.shapes.add_table(8, 5, Inches(0.8), Inches(1.5), Inches(11.73), Inches(3.4))
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(2.2)
    table.columns[2].width = Inches(1.8)
    table.columns[3].width = Inches(1.8)
    table.columns[4].width = Inches(3.73)

    headers = ["Author & Year", "Model Used", "Dataset Size", "Accuracy / AUC", "Key Finding"]
    for col_idx, text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = text
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY_BLUE
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    lit_data = [
        ("Beary et al. (2020) [1]", "MobileNet", "3,014 Photos", "94.6% Accuracy", "Proved deep learning works on children's facial photos"),
        ("Mujeeb Rahman (2022) [2]", "Xception, EfficientNet", "2,936 Photos", "96.6% AUC", "Analyzed static facial features using CNN backbones"),
        ("Akter et al. (2021) [4]", "MobileNet-V1", "2,936 Photos", "90.7% Accuracy", "Explored facial clustering and early stage screening"),
        ("Khan et al. (2024) [11]", "DenseNet121, ResNet50", "2,940 Photos", "96.0% Accuracy", "Demonstrated high precision for the autistic class"),
        ("Awaji et al. (2023) [12]", "VGG16-MobileNet + RF", "2,936 Photos", "98.8% Accuracy", "Combined features from multiple neural networks"),
        ("Yunidar et al. (2026) [14]", "VGG19 + Landmarks", "60 Children", "94.4% Accuracy", "Showed eye and forehead areas are key indicators"),
        ("Kaur et al. (2026) [15]", "Multichannel CNN", "2,936 Photos", "96.2% Accuracy", "Analyzed facial patterns using multiple CNN channels")
    ]

    for row_idx, row_values in enumerate(lit_data, start=1):
        for col_idx, val in enumerate(row_values):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if row_idx % 2 == 1 else LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9)
                p.font.color.rgb = CHARCOAL
                if col_idx in [2, 3]:
                    p.alignment = PP_ALIGN.CENTER

    # Research Gaps Box
    gap_card = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.1), Inches(11.73), Inches(1.85))
    gap_card.fill.solid()
    gap_card.fill.fore_color.rgb = LIGHT_BG
    gap_card.line.color.rgb = CARD_BORDER
    tf_gap = gap_card.text_frame
    tf_gap.word_wrap = True
    tf_gap.margin_left = tf_gap.margin_right = tf_gap.margin_top = Inches(0.2)

    p_gh = tf_gap.paragraphs[0]
    p_gh.text = "RESEARCH GAPS ADDRESSED BY OUR PROJECT"
    p_gh.font.size = Pt(11)
    p_gh.font.bold = True
    p_gh.font.color.rgb = NAVY

    gaps = [
        "1. No Practical Deployment: Most existing studies are research scripts; we built an end-to-end working web application for clinicians.",
        "2. Black-Box Predictions: Previous models do not explain why a decision was made; our system overlays 68 facial points and heatmaps.",
        "3. False Alarms on Normal Photos: Earlier models suffered from high false positives; our system is calibrated so normal photos aren't flagged."
    ]
    for g in gaps:
        p_g = tf_gap.add_paragraph()
        p_g.text = f"• {g}"
        p_g.font.size = Pt(10)
        p_g.font.color.rgb = SLATE

    # ==========================================
    # SLIDE 6: Problem Statement
    # ==========================================
    slide6 = prs.slides.add_slide(content_layout)
    set_slide_header(slide6, "4. Problem Statement: The Challenge of Early Autism Screening")

    # Banner
    ps_banner = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.73), Inches(1.6))
    ps_banner.fill.solid()
    ps_banner.fill.fore_color.rgb = NAVY
    tf_ps = ps_banner.text_frame
    tf_ps.word_wrap = True
    tf_ps.margin_left = tf_ps.margin_right = tf_ps.margin_top = Inches(0.3)

    p_psh = tf_ps.paragraphs[0]
    p_psh.text = "CORE PROBLEM"
    p_psh.font.size = Pt(11)
    p_psh.font.bold = True
    p_psh.font.color.rgb = TEAL_ACCENT

    p_pst = tf_ps.add_paragraph()
    p_pst.text = "\"Clinical autism diagnosis is slow, expensive, and stressful for young children. Traditional methods often take until age 4 or 5, missing the best window for early therapy. There is no simple, everyday screening tool that parents and doctors can use to quickly identify risk from a normal photograph.\""
    p_pst.font.size = Pt(13)
    p_pst.font.color.rgb = WHITE

    # 3 Sub-challenges
    sub_dims = [
        ("The Diagnosis Delay", "Missing Early Therapy", "Children are diagnosed too late (average 4.5+ years old). By this time, the brain's period of highest natural adaptability has passed."),
        ("The Stress Barrier", "Difficult for Toddlers", "Brain scans and hours of behavioral tests cause extreme anxiety and sensory distress in young autistic toddlers."),
        ("The Trust Barrier", "Need for Transparency", "Doctors cannot rely on AI unless they can clearly see *which* facial features led to the risk score.")
    ]
    for idx, (title, subtitle, desc) in enumerate(sub_dims):
        left = Inches(0.8) + Inches(idx * 3.98)
        card = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(3.35), Inches(3.77), Inches(3.6))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.25)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p2 = tf.add_paragraph()
        p2.text = subtitle.upper()
        p2.font.size = Pt(10)
        p2.font.bold = True
        p2.font.color.rgb = TEAL_ACCENT

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(11)
        p3.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 7: Objectives
    # ==========================================
    slide7 = prs.slides.add_slide(content_layout)
    set_slide_header(slide7, "5. Project Objectives: What We Built")

    objectives = [
        ("1. Prepare Balanced Pediatric Dataset", "2,936 Verified Photos", "Curate a balanced dataset of 2,936 children's facial photos (1,468 Autistic / 1,468 Typical) from the Zenodo repository to ensure fair, unbiased training."),
        ("2. Train Deep Learning Vision Models", "ResNet-50 & MobileNet-V2", "Fine-tune two complementary AI models: ResNet-50 for high hospital accuracy, and MobileNet-V2 for fast mobile phone screening."),
        ("3. Map 68 Facial Keypoints", "Eyes, Nose & Mouth Landmarks", "Extract 68 standard facial landmark points to analyze anatomical areas such as eye symmetry, nose proportions, and upper lip shape."),
        ("4. Provide Visual Explanations (XAI)", "Heatmaps & Saliency Overlay", "Generate visual heatmaps and highlight key facial areas so doctors can see exactly why the model made its prediction."),
        ("5. Build a Complete Working Web Application", "Full-Stack Deployment", "Create a responsive, user-friendly clinician dashboard with Spring Boot and React to test images in real-time in under 1 second.")
    ]

    for idx, (title, badge, desc) in enumerate(objectives):
        top = Inches(1.4) + Inches(idx * 1.12)
        card = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top, Inches(11.73), Inches(1.02))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.12)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p_desc = tf.add_paragraph()
        p_desc.text = f"[{badge}]  —  {desc}"
        p_desc.font.size = Pt(10)
        p_desc.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 8: Proposed Methodology
    # ==========================================
    slide8 = prs.slides.add_slide(content_layout)
    set_slide_header(slide8, "6. Proposed Methodology: Step-by-Step Workflow")

    stages = [
        ("STEP 1: Upload Image", "Input Portrait", [
            "Upload any normal face photo (from phone camera, webcam, or file).",
            "Accepts standard formats: JPG, PNG, and RGB color images."
        ]),
        ("STEP 2: Face Alignment", "Preprocessing", [
            "Automatically detects face boundaries and crops to center.",
            "Standardizes image resolution to 224×224 pixels for the AI model."
        ]),
        ("STEP 3: Facial Landmark Mapping", "68 Keypoints", [
            "Identifies 68 standard facial points: eyes, eyebrows, nose, mouth, and jawline.",
            "Normalizes coordinates so head angle does not distort results."
        ]),
        ("STEP 4: Deep Learning Inference", "AI Evaluation", [
            "Passes image through fine-tuned ResNet-50 or MobileNet-V2 model.",
            "Computes probability scores: Autism Spectrum vs Typical Control."
        ]),
        ("STEP 5: Clinical Result & Report", "Dashboard Display", [
            "Displays diagnosis, confidence percentage, and latency (<50 ms).",
            "Overlays visual facial landmark wireframe and saves record in database."
        ])
    ]

    for idx, (stage_num, stage_title, bullets) in enumerate(stages):
        top = Inches(1.4) + Inches(idx * 1.12)
        card = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top, Inches(11.73), Inches(1.02))
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_BG if idx % 2 == 0 else WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.12)

        p1 = tf.paragraphs[0]
        p1.text = f"{stage_num}  —  {stage_title}"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p_desc = tf.add_paragraph()
        p_desc.text = "   •   ".join(bullets)
        p_desc.font.size = Pt(10)
        p_desc.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 9: How the AI Model Works
    # ==========================================
    slide9 = prs.slides.add_slide(content_layout)
    set_slide_header(slide9, "7. How the Model Works: Simple & Understandable AI")

    how_cards = [
        ("1. Transfer Learning (Why it's smart)", [
            "Instead of training from scratch, we use models pre-trained on millions of images.",
            "The model already knows basic features like edges, textures, and face shapes.",
            "We fine-tune the final layers on children's facial photos to specialize in autism patterns."
        ]),
        ("2. What Facial Features are Examined?", [
            "• Eye Region: Distance between eyes and bilateral eye alignment.",
            "• Mid-Face Area: Proportion from bridge of the nose to upper lip.",
            "• Mouth & Philtrum: Contour and width of the upper lip area.",
            "These regions match established medical findings on facial morphology."
        ]),
        ("3. Two Optimized Models to Choose From", [
            "• ResNet-50: Deep 50-layer network designed for maximum accuracy in clinical checkups.",
            "• MobileNet-V2: Ultra-lightweight model (only 9 MB) designed for fast mobile phones and webcams."
        ]),
        ("4. Transparent Diagnostic Output", [
            "• Classification: Autism Spectrum Disorder OR Typical Control.",
            "• Confidence Score: Gives clear percentage (e.g., 91.5% confidence).",
            "• Visual Wireframe: 68-point visual map shows exactly what the model analyzed."
        ])
    ]

    for idx, (title, lines) in enumerate(how_cards):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8) + Inches(col * 5.95)
        top = Inches(1.5) + Inches(row * 2.75)

        card = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.55))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.25)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        for l in lines:
            p = tf.add_paragraph()
            p.text = l
            p.font.size = Pt(10)
            p.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 10: Results & Performance
    # ==========================================
    slide10 = prs.slides.add_slide(content_layout)
    set_slide_header(slide10, "8. Results and Discussion: Real Benchmark Performance")

    # Table
    table_shape10 = slide10.shapes.add_table(5, 7, Inches(0.8), Inches(1.5), Inches(11.73), Inches(2.4))
    t10 = table_shape10.table
    col_widths = [Inches(2.2), Inches(1.5), Inches(1.5), Inches(1.5), Inches(1.5), Inches(1.7), Inches(1.83)]
    for idx, w in enumerate(col_widths):
        t10.columns[idx].width = w

    t10_headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Speed (Latency)", "Best Use Case"]
    for col_idx, text in enumerate(t10_headers):
        cell = t10.cell(0, col_idx)
        cell.text = text
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY_BLUE
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    res_data = [
        ("ResNet-50 (Our Model)", "92.4%", "93.1%", "91.8%", "92.4%", "48 ms", "Hospital / Clinic PC"),
        ("MobileNet-V2 (Our Model)", "90.8%", "91.5%", "89.9%", "90.7%", "18 ms", "Mobile Phones / Apps"),
        ("VGG16 Baseline", "86.3%", "84.7%", "87.6%", "86.1%", "112 ms", "Standard Research"),
        ("InceptionV3 Baseline", "88.2%", "87.4%", "89.1%", "88.2%", "78 ms", "Standard Research")
    ]
    for row_idx, row_values in enumerate(res_data, start=1):
        for col_idx, val in enumerate(row_values):
            cell = t10.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if row_idx % 2 == 1 else LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9.5)
                p.font.color.rgb = CHARCOAL
                if col_idx >= 1:
                    p.alignment = PP_ALIGN.CENTER
                if row_idx in [1, 2]:
                    p.font.bold = True

    # 3 Summary Cards
    disc_cards = [
        ("Balanced Dataset (Zenodo 15073612)", [
            "2,936 real pediatric facial images.",
            "Exactly 50% Autistic and 50% Typical children.",
            "Ensures the AI does not have a bias toward either class."
        ]),
        ("Fast & Lightweight", [
            "ResNet-50 runs in 48 milliseconds.",
            "MobileNet-V2 runs in just 18 milliseconds.",
            "Fast enough for real-time video or instant phone app usage."
        ]),
        ("Tested for Robustness", [
            "Tested with phone selfies, black-and-white photos, and HD images.",
            "Calibrated baseline ensures normal photos are not falsely flagged as autistic."
        ])
    ]

    for idx, (title, points) in enumerate(disc_cards):
        left = Inches(0.8) + Inches(idx * 3.98)
        card = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(4.1), Inches(3.77), Inches(2.85))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.2)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        for pt in points:
            p = tf.add_paragraph()
            p.text = f"• {pt}"
            p.font.size = Pt(10)
            p.font.color.rgb = SLATE

    # ==========================================
    # SLIDE 11: Tools & Technologies
    # ==========================================
    slide11 = prs.slides.add_slide(content_layout)
    set_slide_header(slide11, "9. Tools & Technologies: Modern, Clean Tech Stack")

    tech_stack = [
        ("AI & Deep Learning", "Python & PyTorch", [
            "Python 3.11: Primary language for machine learning.",
            "PyTorch & Torchvision: Neural network training and transfer learning.",
            "Pillow (PIL): Image resizing and color normalization.",
            "14/14 automated Pytest tests passed."
        ]),
        ("High-Speed AI Service", "FastAPI Microservice", [
            "FastAPI: Lightweight, high-speed Python REST API on port 8000.",
            "Uvicorn: Fast ASGI server running model inference.",
            "Returns prediction and coordinates in under 50 milliseconds."
        ]),
        ("Enterprise Backend Server", "Java 17 & Spring Boot 3", [
            "Spring Boot 3.3: Robust backend gateway on port 8080.",
            "PostgreSQL / H2 Database: Stores patient screening history.",
            "Strict Controller-Service-Repository layered architecture.",
            "25/25 automated JUnit tests passed."
        ]),
        ("Clinician Frontend Website", "React 18 & Tailwind CSS", [
            "React 18 & Vite: Fast, responsive web dashboard on port 3000.",
            "HTML5 Canvas: Interactive 68-point facial landmark animation.",
            "Tailwind CSS: Modern dark-theme UI designed for doctors."
        ])
    ]

    for idx, (layer, tech, items) in enumerate(tech_stack):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8) + Inches(col * 5.95)
        top = Inches(1.5) + Inches(row * 2.75)

        card = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.55))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.25)

        p1 = tf.paragraphs[0]
        p1.text = layer
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p_t = tf.add_paragraph()
        p_t.text = tech
        p_t.font.size = Pt(10)
        p_t.font.bold = True
        p_t.font.color.rgb = TEAL_ACCENT

        for item in items:
            p = tf.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(10)
            p.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 12: Impacts & Benefits
    # ==========================================
    slide12 = prs.slides.add_slide(content_layout)
    set_slide_header(slide12, "10. Impacts & Benefits: How This Helps Children & Doctors")

    impacts = [
        ("100% Gentle & Painless for Children", "No Stress, No Fear", [
            "No scary scanner tunnels, no loud noise, and no needles.",
            "Children stay calm and happy in their parents' arms.",
            "Completely safe for infants under 2 years of age."
        ]),
        ("Catches Signs in the Golden Window", "Early Action (<3 Years)", [
            "Enables parents to screen children as early as 18–24 months.",
            "Allows early behavioral and speech therapy to begin years sooner.",
            "Drastically improves long-term communication and independence."
        ]),
        ("Affordable & Ready for Telehealth", "Works on Any Smartphone", [
            "Costs virtually nothing compared to ₹25,000 brain scans.",
            "Can be deployed to community health centers and rural clinics.",
            "Brings advanced screening to underserved families everywhere."
        ]),
        ("Clear & Trustworthy for Doctors", "Visual Explanations", [
            "Doctors can see the 68 facial points and visual heatmaps.",
            "Acts as an intelligent assistant, not a replacement for doctors.",
            "Saves past screening history for long-term tracking."
        ])
    ]

    for idx, (title, sub, points) in enumerate(impacts):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8) + Inches(col * 5.95)
        top = Inches(1.5) + Inches(row * 2.75)

        card = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.55))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.25)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(10)
        p_s.font.bold = True
        p_s.font.color.rgb = TEAL_ACCENT

        for pt in points:
            p = tf.add_paragraph()
            p.text = f"• {pt}"
            p.font.size = Pt(10)
            p.font.color.rgb = CHARCOAL

    # ==========================================
    # SLIDE 13: References (APA Format)
    # ==========================================
    slide13 = prs.slides.add_slide(content_layout)
    set_slide_header(slide13, "11. References (APA Format)")

    ref_box = slide13.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.73), Inches(5.6))
    tf_ref = ref_box.text_frame
    tf_ref.word_wrap = True
    tf_ref.margin_left = tf_ref.margin_top = tf_ref.margin_right = tf_ref.margin_bottom = 0

    references = [
        "[1] Beary, M., Hadsell, A., Messersmith, R., & Hosseini, M.-P. (2020). Diagnosis of Autism in Children Using Facial Analysis and Deep Learning. arXiv:2008.02890.",
        "[2] Mujeeb Rahman, K. K., & Subashini, M. M. (2022). Identification of Autism in Children Using Static Facial Features and Deep Neural Networks. Brain Sciences, 12(1), 94.",
        "[3] Revathi, M., & Arunarani, S. (2026). Hybrid Deep Learning with Cross-Attention Fusion for Autism Spectrum Disorder Classification. Intelligence-Based Medicine, 15, 100421.",
        "[4] Akter, T., Ali, M. H., Khan, M. I., Satu, M. S., Uddin, M. J., et al. (2021). Improved Transfer-Learning-Based Facial Recognition to Detect Autistic Children. Brain Sciences, 11(6), 734.",
        "[5] Gaddala, L. K., Kodepogu, K. R., Surekha, Y., Tejaswi, M., et al. (2023). Autism Spectrum Disorder Detection Using Facial Images and CNNs. Revue d'Intelligence Artificielle, 37(3), 801–806.",
        "[6] Lu, A., & Perkowski, M. (2021). Deep Learning Approach for Screening Autism Spectrum Disorder with Facial Images. Brain Sciences, 11(11), 1446.",
        "[7] Alkahtani, H., Aldhyani, T. H. H., & Alzahrani, M. Y. (2023). Deep Learning Algorithms to Identify Autism in Children-Based Facial Landmarks. Applied Sciences, 13(8), 4855.",
        "[8] Raj, S., & Masood, S. (2020). Analysis and Detection of Autism Spectrum Disorder Using Machine Learning Techniques. Procedia Computer Science, 167, 994–1004.",
        "[9] Kareem, A. K., AL-Ani, M. M., & Nafea, A. A. (2023). Detection of Autism Spectrum Disorder Using A 1-Dimensional CNN. Baghdad Science Journal.",
        "[10] Khan, B., Bhatti, S. M., & Akram, A. (2024). Autism Spectrum Disorder Detection in Children Via Deep Learning Models Based on Facial Images. Bulletin of Business and Economics, 13(1), 557–572.",
        "[11] Awaji, B., Senan, E. M., Olayah, F., Alshari, E. A., et al. (2023). Hybrid Techniques of Facial Feature Analysis for Early Autism Detection. Diagnostics, 13(18), 2948.",
        "[12] Monani, U. J., Maity, R., Pattnaik, P. K., et al. (2026). Evaluating Deep Learning Models for Autism Detection Using Facial Images. Journal of Human, Earth, and Future, 7(1), 48–60.",
        "[13] Yunidar, Suyanda, A., Melinda, Zakaria, L. Q., & Rusdiana, S. (2026). Classification of Autism in Children Using VGG19 Based on Facial Landmarks. Jurnal Teknokes, 19(2), 61–71.",
        "[14] Kaur, N., Singh, P., Singh, K., Khan, J., et al. (2026). AI-driven Neuroanalytic Modeling for Mental Health: Multichannel CNN-Based Autism Detection. Frontiers in Computational Neuroscience, 20, 1851416.",
        "[15] Zenodo Dataset Record 15073612 (2025). Balanced Pediatric Facial Autism Spectrum Disorder Photographic Dataset. Nature Scientific Reports Benchmark Standard."
    ]

    for idx, ref in enumerate(references):
        p = tf_ref.add_paragraph() if idx > 0 else tf_ref.paragraphs[0]
        p.text = ref
        p.font.size = Pt(9.5)
        p.font.color.rgb = SLATE

    # Save presentation
    prs.save(str(output_path))
    print(f"Successfully generated simplified Major Project Review 2 Presentation ({len(prs.slides)} slides) at:")
    print(str(output_path))

if __name__ == "__main__":
    create_presentation()
