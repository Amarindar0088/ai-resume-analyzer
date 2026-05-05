import os
import PyPDF2
from preprocess import clean_text
from skills import skills_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 🔹 Extract text from PDF
def extract_text_from_pdf(file):
    with open(file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# 🔹 Extract skills
def extract_skills(text):
    found_skills = []

    for skill in skills_list:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    return found_skills


# 🔹 Matching function (IMPORTANT)
def match_score(resume_text, job_desc):
    documents = [resume_text, job_desc]

    cv = CountVectorizer()
    matrix = cv.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    return round(similarity * 100, 2)


# 🔹 Job Description
job_description = """
We are looking for a candidate with strong communication skills,
customer service experience, leadership ability, teamwork,
and problem solving skills.
"""


# 🔹 MAIN FLOW
# text = extract_text_from_pdf("resume.pdf")

# cleaned_text = clean_text(text)

# print("\n--- CLEANED TEXT ---\n")
# print(cleaned_text)


# 🔹 Skill extraction
# skills = extract_skills(cleaned_text)

# print("\n--- SKILLS FOUND ---\n")
# print(skills)


# 🔹 Matching score
# score = match_score(cleaned_text, job_description)

# print("\n--- MATCH SCORE ---\n")
# print(str(score) + "%")



folder_path = "resumes"

results = []

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        file_path = os.path.join(folder_path, file)

        text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(text)

        skills = extract_skills(cleaned_text)
        score = match_score(cleaned_text, job_description)

        results.append((file, score, skills))


# 🔹 Sort by score (highest first)
results.sort(key=lambda x: x[1], reverse=True)


print("\n--- CANDIDATE RANKING ---\n")

for i, (file, score, skills) in enumerate(results, start=1):
    print(f"{i}. {file}")
    print(f"   Score: {score}%")
    print(f"   Skills: {skills}\n")