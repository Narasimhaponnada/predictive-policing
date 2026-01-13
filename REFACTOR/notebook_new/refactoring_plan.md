# Crime Forecasting Notebooks Refactoring Plan

## Objective
Refactor the existing crime forecasting notebooks to be cleaner, more readable, and aligned with MSc assessment requirements while preserving the core ML logic.

---

## Key Changes Summary

1. **Style**: Adopt the clean Gemstone notebook style (styled headers, minimal code, clear explanations)
2. **Models**: Use both RandomForest and XGBoost as baseline models, pick best for hyperparameter tuning
3. **Structure**: Follow the 13-step MSc assessment guide
4. **Output**: Create 3 clean notebooks in `notebook_new/` folder
5. **Author**: No author name in headers (just notebook titles)
6. **Baseline**: Mean prediction as simple benchmark (Step 6)
7. **Explainability**: Both SHAP and LIME for comprehensive interpretation

---

## New Notebook Structure

### Notebook 1: `1_Crime_Forecasting_EDA.ipynb`
Covers Assessment Steps 0-5

| Section | Assessment Step | Content |
|---------|-----------------|---------|
| **1. Problem Definition** | Step 0 | Clear problem statement: "Predict crime counts and types by area/time to enable police resource allocation" |
| **2. Decision Context** | Step 1 | Who decides (police commanders), what choice (patrol deployment), consequences of errors |
| **3. ML Paradigm** | Step 2 | Regression for crime counts + proportion-based distribution for crime types |
| **4. Data Understanding** | Step 3 | Dataset description, variables explanation, time span, limitations |
| **5. Data Loading** | - | Load CSV, basic info, shape |
| **6. Missing Values** | Step 4 | Check missing values, explain TIME OCC placeholder issue |
| **7. Duplicates Check** | Step 4 | Check and report duplicates |
| **8. Descriptive Statistics** | Step 5 | Numerical/categorical separation, summary stats |
| **9. Temporal Analysis** | Step 5 | Hourly patterns, weekly patterns, seasonal trends |
| **10. Spatial Analysis** | Step 5 | Area-based crime distribution, geographic patterns |
| **11. Crime Type Analysis** | Step 5 | Top crime types, type-specific patterns |
| **12. Correlation Analysis** | Step 5 | Feature relationships, heatmap |
| **13. EDA Summary** | Step 5 | Key findings that inform model choice |

### Notebook 2: `2_Crime_Forecasting_Model.ipynb`
Covers Assessment Steps 6-10

| Section | Assessment Step | Content |
|---------|-----------------|---------|
| **1. Data Preprocessing** | Step 4 | Time imputation, feature extraction, crime categorization |
| **2. Data Aggregation** | Step 4 | Transform to time windows, create target variable |
| **3. Feature Engineering** | Step 4 | Lag features, rolling averages |
| **4. Train-Test Split** | Step 8 | Chronological 80/20 split |
| **5. Data Pipeline** | Step 8 | ColumnTransformer with preprocessing |
| **6. Baseline Model** | Step 6 | Simple mean/median prediction as benchmark |
| **7. Model Selection** | Step 7 | Justify regression models for crime count prediction |
| **8. Baseline Models Training** | Step 8 | Train RandomForest and XGBoost with default settings |
| **9. Baseline Comparison** | Step 9 | Compare MAE, RMSE, R2 for both models |
| **10. Best Model Selection** | Step 7 | Pick best performer with justification |
| **11. Hyperparameter Tuning** | Step 8 | RandomizedSearchCV on best model |
| **12. Final Model Evaluation** | Step 9 | Comprehensive metrics tied to decision impact |
| **13. Stage 2: Crime Type Distribution** | Step 8 | Proportion-based distribution method |
| **14. Two-Stage Evaluation** | Step 9 | Combined system performance |
| **15. Feature Importance** | Step 10 | What variables matter most |

### Notebook 3: `3_Crime_Forecasting_Explainability.ipynb`
Covers Assessment Steps 10-13

| Section | Assessment Step | Content |
|---------|-----------------|---------|
| **1. Model Loading** | - | Load trained model and preprocessor |
| **2. SHAP Analysis** | Step 10 | Global feature importance, summary plots |
| **3. LIME Analysis** | Step 10 | Local instance-level explanations |
| **4. Prediction Examples** | Step 10 | Input-output walkthrough for stakeholders |
| **5. Model Limitations** | Step 11 | Data gaps, assumptions, failure modes |
| **6. Ethics & Responsibility** | Step 12 | Bias risks, transparency, responsible deployment |
| **7. Conclusion & Reflection** | Step 13 | Was ML right? What would change? Lessons learned |

---

## Detailed Implementation

### Style Guidelines (from Gemstone notebooks)

1. **Section Headers**: Use styled HTML headers
```html
<p style="padding:10px;background-color:#87CEEB;margin:10;color:#000000;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 10px 10px;overflow:hidden;font-weight:50">Section Title</p>
```

2. **Subsections**: Use `###` markdown headers
3. **Code Comments**: Minimal inline comments, let markdown explain
4. **Outputs**: Concise, no excessive print statements
5. **Visualizations**: One purpose per chart, explained in markdown

### Model Training Approach

**Baseline Comparison (both with default settings):**
```python
models = {
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "XGBoost Regressor": XGBRegressor(random_state=42)
}
```

**Evaluation Function:**
```python
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    return mae, rmse, r2
```

**Selection Criteria:**
- Compare test set R2 scores
- Consider train-test gap (overfitting check)
- Pick model with best generalization

**Hyperparameter Tuning (on selected model):**
- If XGBoost wins: tune n_estimators, max_depth, learning_rate, subsample
- If RandomForest wins: tune n_estimators, max_depth, min_samples_split

### Core Logic to Preserve

1. **Time Imputation**: Smart redistribution of 12:00 PM placeholder times
2. **Data Aggregation**: Group by (area, date, time_block) with crime counts
3. **Feature Set**: 12 features including lags and rolling averages
4. **Two-Stage Architecture**: Stage 1 (total count) + Stage 2 (type distribution)
5. **Chronological Split**: 80/20 time-based split (no shuffle)
6. **Evaluation Metrics**: MAE, RMSE, R2, within-range accuracy

---

## Files to Create

| File | Location |
|------|----------|
| `1_Crime_Forecasting_EDA.ipynb` | `notebook_new/` |
| `2_Crime_Forecasting_Model.ipynb` | `notebook_new/` |
| `3_Crime_Forecasting_Explainability.ipynb` | `notebook_new/` |

---

## MSc Assessment Steps Coverage Matrix

| Step | Description | Notebook | Section |
|------|-------------|----------|---------|
| 0 | Define the Problem | Notebook 1 | Section 1 |
| 1 | Define the Decision | Notebook 1 | Section 2 |
| 2 | Choose ML Paradigm | Notebook 1 | Section 3 |
| 3 | Understand the Data | Notebook 1 | Section 4 |
| 4 | Data Preparation | Notebook 1 (check) + Notebook 2 (transform) | Sections 6-7, 1-3 |
| 5 | Exploratory Analysis | Notebook 1 | Sections 8-12 |
| 6 | Establish Baseline | Notebook 2 | Section 6 |
| 7 | Model Selection | Notebook 2 | Sections 7, 10 |
| 8 | Model Training | Notebook 2 | Sections 8, 11, 13 |
| 9 | Evaluation Metrics | Notebook 2 | Sections 9, 12, 14 |
| 10 | Interpretation | Notebook 2 + Notebook 3 | Section 15, Sections 2-4 |
| 11 | Limitations | Notebook 3 | Section 5 |
| 12 | Ethics & Responsibility | Notebook 3 | Section 6 |
| 13 | Conclusion & Reflection | Notebook 3 | Section 7 |

---

## Verification Plan

1. **Run EDA Notebook**: Verify all visualizations render, data loads correctly
2. **Run Model Notebook**: Verify both models train, metrics calculated, best model selected
3. **Run Explainability Notebook**: Verify SHAP analysis works, conclusions align with findings
4. **Check Assessment Alignment**: Review each of 13 steps is clearly addressed
5. **Style Check**: Verify consistent formatting, no excessive output, clean presentation
