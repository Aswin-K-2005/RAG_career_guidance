import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.enums import TA_CENTER
import os
import re

# --- Configuration ---
# This file name MUST match the one expected by your RAG system (app.py)
OUTPUT_PDF_FILE = "skillsense - pdf.pdf"
JOBS_DATA_PATH = "data/jobs/job_dataset.csv"
COURSES_DATA_PATH = "data/courses/all_courses.csv"

# --- PDF Generation Styles ---
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleStyle', alignment=TA_CENTER, fontSize=24, spaceAfter=20, fontName='Helvetica-Bold'))
# Fixed syntax: changed ':' to '=' in ParagraphStyle definitions
styles.add(ParagraphStyle(name='Heading1', fontSize=16, spaceBefore=20, spaceAfter=10, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='Heading2', fontSize=12, spaceBefore=10, spaceAfter=5, fontName='Helvetica-Bold'))
# Basic style for content that RAG will read
styles.add(ParagraphStyle(name='BodyText', fontSize=10, spaceAfter=6, fontName='Helvetica'))
styles.add(ParagraphStyle(name='RequirementsHeader', fontSize=14, spaceBefore=15, spaceAfter=5, fontName='Helvetica-Bold', textColor='#CC0000'))
styles.add(ParagraphStyle(name='RequirementsBody', fontSize=10, spaceAfter=10, fontName='Helvetica-Oblique'))

def load_data():
    """Loads the pre-processed Job and Course data from the paths specified."""
    try:
        # Load the data files created in your data processing steps
        jobs = pd.read_csv(JOBS_DATA_PATH)
        courses = pd.read_csv(COURSES_DATA_PATH)
        print(f"Loaded {len(jobs)} jobs and {len(courses)} courses.")
        return jobs, courses
    except FileNotFoundError as e:
        print(f"Error: Data file not found at {e.filename}. Please ensure you have run the data processing steps first.")
        # Return empty DataFrames if files are missing to prevent crash
        return pd.DataFrame(), pd.DataFrame()

def clean_skills_for_pdf(skill_list_str):
    """Cleans the string representation of skills from the CSVs for PDF text."""
    if pd.isna(skill_list_str):
        return 'None specified.'
    # Remove bracket/quote artifacts and join items with commas
    cleaned = re.sub(r'[\{\}\[\]"\']', '', str(skill_list_str))
    return cleaned.strip()

def create_pdf_content(jobs_df, courses_df):
    """Generates the flowable objects (Paragraphs, Spacers) for the PDF document."""
    story = []

    # Title Page/Header
    story.append(Paragraph("SkillSense Comprehensive Knowledge Base", styles['TitleStyle']))
    story.append(Paragraph(
        "This document contains structured, textual data on job roles and available online courses. Its primary purpose is to serve as a **context source** for the SkillSense RAG Chatbot.", 
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))

    # --- NEW: Static System Requirements and Instructions ---
    story.append(Paragraph("System Requirements and Instructions", styles['RequirementsHeader']))
    
    requirements_text = """
    This section dictates the rules for the SkillSense RAG system:
    1. The RAG system **must only answer questions** based on the content available within the following Job Descriptions (SECTION 1) or Course Catalog (SECTION 2). 
    2. If the user's question cannot be factually answered by the text in this document, the system must politely state that the information is outside the scope of its current knowledge base.
    3. When providing recommendations, the system must prioritize matching the user's input skills to the listed 'Primary Skills Required' or 'Key Skills Taught' found in the respective sections.
    4. All responses must be concise, helpful, and reference the source sections (Job or Course) where appropriate.
    """
    story.append(Paragraph(requirements_text.replace('\n', '<br/>'), styles['RequirementsBody']))
    story.append(Spacer(1, 24))


    # --- 1. Job Descriptions Section ---
    story.append(Paragraph("SECTION 1: Job Descriptions and Requirements", styles['Heading1']))

    # Iterate over jobs (only taking the first 20 entries for a manageable example PDF)
    # NOTE: Assuming 'job_dataset.csv' columns (Title, Skills, Keywords, Description) are correct.
    for _, job in jobs_df.head(20).iterrows():
        job_info = []
        
        skills = clean_skills_for_pdf(job.get('Skills', 'N/A'))
        keywords = clean_skills_for_pdf(job.get('Keywords', 'N/A'))
        
        # Create a single, rich paragraph for each job for easier RAG chunking
        job_paragraph = f"**JOB ROLE: {job.get('Title', 'Unknown Job')}**."
        job_paragraph += f" Primary Skills Required: {skills}."
        job_paragraph += f" Associated Keywords/Focus: {keywords}."
        
        job_info.append(Paragraph(job_paragraph, styles['Heading2']))
        
        # Add any description if available
        description = job.get('Description', 'Detailed description not provided in the source job dataset.')
        job_info.append(Paragraph(f"Description Summary: {description[:200]}...", styles['BodyText']))
        
        job_info.append(Spacer(1, 6))
        
        # Use KeepTogether to ensure the job entry is not split across pages awkwardly
        story.append(KeepTogether(job_info))
        story.append(Spacer(1, 12))


    # --- 2. Course Catalog Section ---
    story.append(Paragraph("SECTION 2: Online Course Catalog", styles['Heading1']))
    
    # Iterate over courses (only taking the first 20 entries for a manageable example PDF)
    for _, course in courses_df.head(20).iterrows():
        course_info = []
        
        skills = clean_skills_for_pdf(course.get('skills', 'N/A'))
        rating = course.get('rating', 'N/A')
        link = course.get('link', 'N/A')

        # Create a single, highly-structured paragraph for the course
        course_paragraph = f"**COURSE TITLE: {course.get('title', 'Unknown Course')}**."
        course_paragraph += f" Offered by: {course.get('institution', 'N/A')} on the {course.get('platform', 'N/A')} platform."
        course_paragraph += f" Target Level: {course.get('level', 'N/A')}."
        course_paragraph += f" Key Skills Taught: {skills}."
        
        # Adding Rating and Link information from the CSV
        if pd.notna(rating) and rating != 'N/A':
             course_paragraph += f" User Rating: {rating}."
        
        if pd.notna(link) and link != 'N/A' and link:
             course_paragraph += f" Direct Link: {link}."
        
        course_info.append(Paragraph(course_paragraph, styles['Heading2']))
        course_info.append(Spacer(1, 6))
        
        story.append(KeepTogether(course_info))
        story.append(Spacer(1, 12))
        
    story.append(Paragraph(f"Total entries included in this PDF: {len(jobs_df.head(20))} Jobs and {len(courses_df.head(20))} Courses.", styles['BodyText']))

    return story

def generate_pdf(jobs_df, courses_df):
    """Creates the PDF file."""
    # Ensure the directory exists
    if not os.path.exists(os.path.dirname(OUTPUT_PDF_FILE)):
         os.makedirs(os.path.dirname(OUTPUT_PDF_FILE), exist_ok=True)
         
    doc = SimpleDocTemplate(OUTPUT_PDF_FILE, pagesize=letter)
    story = create_pdf_content(jobs_df, courses_df)
    
    print(f"Starting PDF generation for '{OUTPUT_PDF_FILE}'...")
    try:
        doc.build(story)
        print(f"Successfully created PDF at: {OUTPUT_PDF_FILE}")
    except Exception as e:
        print(f"Failed to build PDF. Do you have 'reportlab' installed? Error: {e}")

if __name__ == "__main__":
    jobs_df, courses_df = load_data()
    if len(jobs_df) > 0 and len(courses_df) > 0:
        generate_pdf(jobs_df, courses_df)
    else:
        print("PDF generation was skipped due to empty or missing data. Check your data preparation steps.")
