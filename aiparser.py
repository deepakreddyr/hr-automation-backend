from openai import OpenAI
from scrapper import scrape
import os 
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
def get_candidate_details(data,jd1,skills):
    
    client=OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""
    Evaluate a batch of candidate resumes against a given job description (JD) to identify and shortlist qualified candidates. For each candidate, calculate a match score from 0 to 100 based on relevance to the JD. Only candidates with a score above 70 should be shortlisted. Provide output in structured JSON format for each candidate, with detailed fields for shortlisted ones and a rejection reason for others.

- Match Score Calculation: Analyze each candidate's resume against the JD to calculate a match score in the range of 0 to 100.
- Data Extraction: For candidates with a match score greater than 70, extract key fields from their resumes.
- Conditional Output: Vary the structure of the JSON output based on the match score for each resume.

# Steps

1. **Calculate Match Score**: 
   - Compare the candidates' resumes with the {jd1} to determine a match score within 0-100 for each resume.
   
2. **Conditional Logic**:
   - If a match score is greater than 70, proceed to extract detailed information from the resume.
   - If a match score is 70 or below, skip data extraction and generate a rejection reason.

3. **Data Extraction** (for match scores > 70):
   - Extract and include the following fields in the JSON output for each candidate with a high score:
     - `match_score`
     - `name`
     - `phone`
     - `email`
     - `job_summary`
     - `experience in each skill` as mentioned in {skills}
     - `total_experience` in years and months
     - `relevant_experience` in specified skills in years and months

4. **Rejection Cases** (for match scores ≤ 70):
   - Output a JSON containing:
     - `match_score`
     - `reason_to_reject`

5. **The number of candidates resume results should strictly be equal to the number of candidates resume input(IMPORTANT).**
# Output Format

Produce the output in JSON format for each resume:
- **If match_score > 70 for a resume**:
  {{
    "match_score": [value],
    "name": "[Name]",
    "phone": "[Phone Number]",
    "email": "[Email]",
    "job_summary": "[Summary]",
    "experience_in_skills": {{
      "[Skill1]": "[Duration]",
      "[Skill2]": "[Duration]"
    }},
    "total_experience": "[Years and Months]",
    "relevant_experience": "[Years and Months]"
  }}
"""


    response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            "text": system_prompt
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "input_text",
            "text": data
            }
        ]
        },
    ],
    text={
        "format": {
        "type": "text"
        }
    },
    reasoning={},
    tools=[],
    temperature=0.5,
    max_output_tokens=2048,
    top_p=1,
    store=True
    )

    try:
        raw_output = response.output[0].content[0].text.strip()

        if not raw_output:
            print("❌ ERROR: Empty model output")
            return []

        # Try parsing the raw output
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            print("⚠️ WARNING: Output is not a valid JSON list or object.")
            print("🔎 Raw output:", raw_output)
            return []

        # If it's a single object, wrap it in a list
        if isinstance(parsed_output, dict):
            print("ℹ️ INFO: Wrapping single JSON object in list.")
            parsed_output = [parsed_output]

        # Final check: Ensure it's a list
        if not isinstance(parsed_output, list):
            print("❌ ERROR: Parsed output is not a list.")
            return []

        print("✅ Parsed JSON:", parsed_output)
        return parsed_output

    except (AttributeError, IndexError) as e:
        print("❌ ERROR: Unexpected response format:", e)
        print("🧾 Full response object:", response)
        return []

    


jd ='''Data Analyst
Job Description
Overview:
We are seeking a talented and detail-oriented Data Analyst to join our dynamic team. As a Data Analyst,
you will be responsible for interpreting data, analyzing results, and providing actionable insights to drive
informed decision-making across the agency. You will work closely with stakeholders to understand their
data needs, develop analytical solutions, and present findings in a clear and concise manner.
Responsibilities
Data Collection and Processing
• Extract, transform, and load (ETL) data from various sources.
• Clean and preprocess data to ensure accuracy and consistency.
• Develop scripts and workflows to automate data collection and processing tasks.
Data Analysis and Interpretation:
• Perform exploratory data analysis to uncover trends, patterns, and anomalies.
• Apply statistical and analytical techniques to derive insights from complex datasets.
• Conduct hypothesis testing and predictive modeling to support business objectives.
Data Visualization and Reporting
• Create visually appealing and interactive dashboards, reports, and presentations.
• Communicate findings and recommendations to stakeholders using data visualization tools.
• Collaborate with cross-functional teams to design and deliver actionable insights.
Data Quality Assurance
• Validate data accuracy, completeness, and integrity.
• Identify and address data quality issues and discrepancies.
• Implement data quality controls and monitoring mechanisms.
Business Intelligence and Decision Support
• Partner with business units to define key performance indicators (KPIs) and metrics.
• Analyze business processes and operations to identify opportunities for improvement.
• Provide decision support through ad-hoc analysis and scenario modeling.
Continuous Learning and Development
• Stay abreast of industry trends, best practices, and emerging technologies in data analytics.
• Participate in training programs and professional development opportunities.
• Share knowledge and insights with colleagues to foster a culture of continuous learning.
Requirements:
• Bachelor's degree in Statistics, Mathematics, Computer Science, Economics, or related field.
Master's degree preferred.
• Proven experience in data analysis, business intelligence, or related roles.
• Proficiency in data analysis tools and programming languages (e.g., SQL, Python, R, etc.).
• Strong analytical and problem-solving skills with attention to detail.
• Excellent communication and presentation abilities.
• Ability to work independently and collaboratively in a fast-paced environment.
• Experience with data visualization tools (e.g., Tableau, Power BI, etc.) is a plus.
• Knowledge of machine learning and data mining techniques is desirable.'''


# jd1='''MERN Stack Developer
# Job Description

# Overview:
# We are seeking a skilled and motivated MERN Stack Developer to join our growing development team. As a MERN Stack Developer, you will be responsible for designing, developing, and maintaining scalable web applications using MongoDB, Express.js, React.js, and Node.js. You will work closely with product managers, designers, and backend teams to deliver high-quality solutions that enhance our digital products and user experience.

# Responsibilities
# Full-Stack Web Development
# Develop modern, responsive, and interactive web applications using React.js.

# Build robust backend services and RESTful APIs using Node.js and Express.js.

# Design and manage schemas and queries in MongoDB.

# System Architecture and Design
# Collaborate with cross-functional teams to gather and analyze requirements.

# Architect efficient, reusable, and scalable front-end and back-end systems.

# Ensure application performance, responsiveness, and scalability.

# Code Quality and Maintenance
# Write clean, maintainable, and well-documented code.

# Conduct code reviews, unit testing, and integration testing.

# Identify and fix bugs and performance bottlenecks.

# DevOps and Deployment
# Participate in the CI/CD pipeline and deployment processes.

# Ensure applications are secure, stable, and well monitored post-deployment.

# Optimize applications for maximum speed and scalability in production.

# Collaboration and Communication
# Work closely with UI/UX designers to bring wireframes and mockups to life.

# Communicate technical information clearly to both technical and non-technical stakeholders.

# Provide ongoing support and improvements to existing applications.

# Continuous Learning and Improvement
# Stay up-to-date with the latest industry trends and technologies in web development.

# Explore and suggest new tools and technologies to improve development workflow.

# Contribute to team knowledge-sharing and best practices.

# Requirements
# Bachelor’s degree in Computer Science, Engineering, or related field. Master’s preferred.

# Proven experience in full-stack development using the MERN stack (MongoDB, Express.js, React.js, Node.js).

# Strong understanding of JavaScript (ES6+), HTML, CSS, and RESTful API development.

# Experience with state management libraries like Redux or Context API.

# Familiarity with Git, version control, and collaborative development workflows.

# Experience with deployment and cloud platforms (e.g., AWS, Heroku, Vercel) is a plus.

# Knowledge of authentication, authorization, and security best practices.

# Excellent problem-solving, debugging, and communication skills.

# Ability to work both independently and in a team-oriented, agile environment.

# '''

data='''



NISHA KUMARI +91-9709352282 # nisha06k@gmail.com ï LinkedIn § GithubEducationSRM University, Chennai 2017 - 2021B.Tech in Information Technology 7.01/10 CGPASaraswati Vidya Mandir, Dugda 2017Higher Secondary 62.3/10 PctSaraswati Vidya Mandir, Dugda 2015Secondary 5.8/10 CGPACoursework/SkillsSQL, Power BI Service, Power Query, SQL Server, Excel, Data Visualization, Data Cleaning, Data ModellingWork ExperienceDXC Technology June 2022 – July 2023Associate Professional Technical SupportSupporting clients with technical issues, including on-site inspections.Fixing bugs by applying updates, upgrades, and software patches.Provided Technical support for clients Remotely and solved the issue. maintaining and fixing any error, or failureGood understanding of data querying / SQL.Concentrix Services Ind Pvt Ltd June 2021 – June 2022Technical Support AssociateHandling users over IB and OB calls. Diagnosing and dispatching faulty parts for consumer machines. Tracking and followingup with partner’s onsite service and assisting users with remote support for S/W-related issues. Taking care of Windowsinstallation and configuration.ProjectsCOVID19 Project Source CodeThe aim of the project was good predictionData Cleaning: Deleting rows with null values and columns not needed for this analysis, and converting data types,Encoding.Data Validation and Exploration: I used Python for Data Exploration(Barchart and Heatmap), separating dependent andindependent variables And splitting the data into train and testAccuracy: Used Machine learning for model training, evaluated data using test data, Feature Scaling and LogisticRegression, Linear Regression, Decision Tree, and Random Forest.SQL: SQL is also used to find some values.Result: The highest accuracy from the Decision Tree is 84.Technical Skills• SQL• Python• Data Analysis• Statistic• Data Visualization• Machine LearningCertificationsData Science Bootcamp August 2023 – February 2024OdinSchoolSQL Udemy
'''

# get_candidate_details(data,jd,"python,sql")

res = []
prof = []

def shortlist_candidates(candidates, required_skills):
    if isinstance(required_skills, str):
        required_skills = [skill.strip().lower() for skill in required_skills.split(",")]
    else:
        required_skills = [skill.strip().lower() for skill in required_skills]

    shortlisted_indices = []

    for idx, candidate in enumerate(candidates):
        candidate_text = " ".join([str(x) for x in candidate]).lower()
        matched_skills = [skill for skill in required_skills if skill in candidate_text]

        if len(matched_skills) == len(required_skills):
            shortlisted_indices.append(idx+1)

    return shortlisted_indices

def scrape(data):
    global res, prof
    res = []  # Reset global list

    for i in data.split("\n"):
        res.append(i.strip())

    sliced_data = res[46:len(res)-34]

    final = []
    temp = []

    for i in sliced_data:
        if "active" in i.lower():
            final.append(temp)
            temp = []
        elif i in ["View phone number", "Call candidate", "Verified phone & email", "\n", ""]:
            continue
        else:
            temp.append(i)

    return final




from openai import OpenAI

def get_questions(jd):
    client = OpenAI()
    prompt=f"""Generate HR-level interview questions from a provided job description to confirm a candidate's experience and qualifications. Focus on evaluating demonstrated work related to the key skills and experiences mentioned within the job description.\n\n[Frame questions based on the provided job description template.]\n\n# Steps\n\n1. Analyze the provided job description to extract key skills, experience, and attributes required for the role.\n2. Formulate questions aimed at confirming the candidate's direct experience and work related to these specific criteria from the job description.\n3. Ensure questions are specific to the job description details, yet open enough to evoke detailed responses.\n\n# Input\n\n- {jd}: A structured job description from which key skills, experiences, and qualifications will be identified.\n\n# Output Format\n\nThe output should be a structured list of HR-level interview questions.\n\n- Example Questions:\n  - \"Can you detail your experience working with {{key_skill_1}} as described in the job description?\"\n  - \"The job description mentions experience in {{key_area}}. Can you share a specific project in which you utilized these skills?\"\n  - \"Can you provide examples of how you have demonstrated {{required_attribute}} in previous roles?\"\n\n# Notes\n\n- Tailor questions to verify the candidate's past work and experience directly related to the job description provided.\n- The aim is to assess the candidate's specific capabilities aligning with the job requirements described. (Do not generate more than 3 questions and also make sure that the questions are shorter which will also have shorter answers)"""
    response = client.responses.create(
    model="gpt-4.1-nano",
    input=[
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            "text": prompt
            }
        ]
        }
    ],
    text={
        "format": {
        "type": "text"
        }
    },
    reasoning={},
    tools=[],
    temperature=1,
    max_output_tokens=2048,
    top_p=1,
    store=True
    )

    print(response.output[0].content[0].text)
    raw_output = response.output[0].content[0].text
    return raw_output

# get_questions(jd)