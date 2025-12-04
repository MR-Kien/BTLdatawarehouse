BTLdatawarehouse/
│
├─ Job 1.kjb # Master ETL job orchestrating all transformations
├─ Transformation\_\*.ktr # ETL pipelines (Dim, Fact, ML data)
│
├─ UI/
│ ├─ app.py # Flask/Streamlit app to run recommendation
│ ├─ bayes_recommendation_model.pkl
│ ├─ requirements.txt
│
└─ README.md

# How to run locally

git clone https://github.com/<yourrepo>/BTLdatawarehouse.git
cd BTLdatawarehouse
cd UI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
