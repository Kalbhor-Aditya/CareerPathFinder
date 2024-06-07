import nltk 
import pickle 
import re 
import streamlit as st


nltk.download('punkt')
nltk.download('stopwords')


clf=pickle.load(open('clf_GB.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

def cleanResume(txt):
  cleantxt=re.sub('http\S+\s',' ',txt)
  cleantxt=re.sub('RT|cc','.',cleantxt)
  cleantxt=re.sub('#\S+\s',' ',cleantxt)
  cleantxt=re.sub('@\S+',' ',cleantxt)
  cleantxt=re.sub('[%s]' % re.escape("""!*"#$%&'()"+,-./:;<=>=?@[\]^_{|}~"""),' ',cleantxt)
  cleantxt=re.sub(r"[^\x00-\x7f]",' ',cleantxt)
  cleantxt=re.sub('\s+',' ',cleantxt)

  return cleantxt

def main():
    st.title("🔎CareerPath Predictor")
    st.subheader("Upload your resume, and let's predict the most suitable career path for you!🚀")
    uploaded_file = st.file_uploader("Upload Resume",type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')

        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')

        cleaned_resume=cleanResume(resume_text)
        cleaned_resume=tfidf.transform([cleaned_resume])
        clf.predict(cleaned_resume)

        prediction_id=clf.predict(cleaned_resume)[0]

        #st.write(prediction_id)

        category_mapping={
            15:"Java Developer",
            23:"Testing",
            8: "Devops Engineer",
            20: "Python Developer",
            24:"Web Designing",
            12: "HR",
            13: "Hadoop",
            3:"Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
            }

        category_name=category_mapping.get(prediction_id,"Unkown")

        st.subheader("🎯Prediction Result: ")
        #st.write(category_name)
        st.success(f"The most suitable career path for you is: **{category_name}**")

if __name__ == "__main__":
    main()
