from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ------------------ PDF TEXT EXTRACTION ------------------
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# ------------------ SKILL EXTRACTION ------------------
def extract_skills(text):
    skills_list = [
        "python", "java", "c++", "sql", "html", "css", "javascript",
        "machine learning", "data analysis", "flask",
        "communication", "teamwork", "problem solving"
    ]

    found = []
    for skill in skills_list:
        if skill in text.lower():
            found.append(skill)

    return found


# ------------------ MATCH SCORE ------------------
def match_score(resume, job_desc):
    documents = [resume, job_desc]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(score[0][0] * 100, 2)


# ------------------ SKILL GAP ------------------
def skill_gap(found_skills):
    required_skills = [
        "python", "flask", "machine learning",
        "data analysis", "problem solving",
        "communication", "teamwork"
    ]

    missing = []
    for skill in required_skills:
        if skill not in found_skills:
            missing.append(skill)

    return missing


# ------------------ FEEDBACK ------------------
def give_feedback(score):
    if score >= 70:
        return "Great match! Your resume is strong."
    elif score >= 40:
        return "Good match, but you can improve."
    else:
        return "Low match. Try improving your resume with relevant skills."


# ------------------ ROUTE ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    skills = []
    score = None
    missing_skills = []
    feedback = ""
    job_desc = ""

    if request.method == "POST":
        file = request.files["resume"]
        job_desc = request.form.get("job_desc")

        if file and job_desc:
            text = extract_text(file)

            skills = extract_skills(text)
            score = match_score(text, job_desc)
            missing_skills = skill_gap(skills)
            feedback = give_feedback(score)

    return render_template(
        "index.html",
        skills=skills,
        score=score,
        missing_skills=missing_skills,
        feedback=feedback,
        job_desc=job_desc
    )


if __name__ == "__main__":
    app.run(debug=True)