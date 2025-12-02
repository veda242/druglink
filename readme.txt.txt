KnowDDI – Backend Package (Lightweight Version for Submission)
-------------------------------------------------------------

This folder contains the cleaned dataset, the trained Drug–Drug Interaction (DDI) model, and the necessary Python scripts. It is a lightweight version (<100 MB) so it can be easily shared and executed.

-------------------------------------------------------------
1. Folder Structure
-------------------------------------------------------------

final/
│   README.txt
│   requirements.txt
│
├── data/
│      dailymed_interactions.csv        → Processed interaction dataset
│
├── models/
│      ddi_baseline_model.joblib        → Trained TF-IDF + Logistic Regression model
│
└── scripts/
       preprocess_dailymed.py           → Extracts interactions from DailyMed XML files
       dailymed_dataset.py              → Dataset utilities (parsing, cleaning)
       train_baseline.py                → Code used to train the logistic baseline


-------------------------------------------------------------
2. What This Package Does
-------------------------------------------------------------

✔ Preprocesses DailyMed XML files into a structured CSV dataset  
✔ Trains a baseline drug–drug interaction (DDI) classifier  
✔ Saves the DDI model as a reproducible Joblib file  
✔ Can be executed on Google Colab without installing large dependencies  
✔ Ready for integration with LLM-based explanation model (future work)

-------------------------------------------------------------
3. How to Run the Backend
-------------------------------------------------------------

1. Upload this `final/` folder to Google Colab.
2. Install dependencies:

    ```python
    !pip install -r requirements.txt
    ```

3. Load the trained model:

    ```python
    import joblib
    model = joblib.load("models/ddi_baseline_model.joblib")
    ```

4. Make predictions:

    ```python
    sample = ["ketoconazole and atorvastatin interaction"]
    pred = model.predict(sample)
    print(pred)
    ```

-------------------------------------------------------------
4. Notes for Evaluation
-------------------------------------------------------------

• This is the lightweight submission version.  
• Full DailyMed XML dataset (~3 GB) is not included.  
• Full training pipeline can be re-run using the scripts in the folder.  
• The model currently uses TF-IDF + Logistic Regression.  
• LLM-based explanation module will be integrated after review.

-------------------------------------------------------------
5. Contact Info (if needed)
-------------------------------------------------------------

Student: (Your Name)  
Project: KnowDDI – Interpretable Drug–Drug Interaction Prediction  
Guide: (Professor Name)

-------------------------------------------------------------
