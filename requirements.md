# LA Crime Forecasting System - Project Requirements

## Project Overview
This project aims to build a **crime forecasting system** that predicts future crime activity in Los Angeles. The system takes an area and time window as input and forecasts:
1. **How many crimes** are expected to occur
2. **What types of crimes** are most likely

This enables **proactive police deployment** rather than reactive response.

## Dataset Information
- **Source**: Crime_Data_from_2020_to_Present.csv
- **Official Source**: https://catalog.data.gov/dataset/crime-data-from-2020-to-present
- **Coverage**: Los Angeles crime incidents from 2020 onwards
- **Available Data Features**:
  - **Temporal**: Date Reported, Date Occurred, Time Occurred
  - **Geographic**: Area codes, reporting districts, coordinates (LAT/LON), location descriptions
  - **Crime Classification**: Crime codes, descriptions, Part 1-2 classification
  - **Incident Details**: DR numbers, MO codes, premise information
  - **Demographics**: Victim age, sex, descent when available
  - **Weapons**: Weapon codes and descriptions when applicable
  - **Status**: Investigation status information
- **Purpose**: Crime pattern analysis and predictive modeling

## Project Objectives

### Primary Objective: Crime Forecasting System

**INPUT**: 
- Area/District (e.g., "Central LA", "Hollywood")
- Time Window (e.g., "January 15, 2026, 6-9 PM")

**OUTPUT**:
- **Crime Count Prediction**: How many crimes are expected?
- **Crime Type Distribution**: What types of crimes are most likely?

### Use Cases:
1. **Police Deployment Planning** - Allocate officers based on predicted crime activity
2. **Resource Optimization** - Position patrol units in high-risk areas during high-risk times
3. **Budget Planning** - Forecast crime trends for staffing decisions
4. **Community Safety** - Proactive crime prevention rather than reactive response

### Secondary Objectives:

#### 1. Temporal Pattern Analysis
- Peak crime hours/days identification
- Seasonal trends and patterns
- Holiday and special event impact

#### 2. Geographic Risk Assessment
- High-risk area identification
- District-level crime forecasting
- Hotspot prediction and evolution

#### 3. Crime Type Distribution
- Predict probability of different crime types
- Specialized unit deployment (e.g., gang unit, drug enforcement)
- Resource-specific allocation (e.g., property crime vs violent crime teams)

## Technical Requirements

### 4. Machine Learning Approach: Crime Forecasting

#### 4.1 Problem Type: **REGRESSION** (not Classification)

**Correct Approach:**
- **Task**: Predict crime COUNT for given area + time window
- **Model Type**: XGBoost Regressor (XGBRegressor)
- **Input**: Area, Date, Time Block, Historical Crime Counts
- **Output**: Predicted number of crimes (continuous value)

**Why Regression?**
- We're predicting "how many" (count), not "which category"
- Forecasting future crime activity, not classifying existing crimes
- Enables resource allocation based on expected volume

#### 4.2 Why XGBoost is EXCELLENT for Crime Forecasting

##### **XGBoost Regressor** ⭐⭐⭐⭐⭐

**Perfect Fit for Crime Forecasting:**

✅ **Regression Capability**
- XGBRegressor predicts continuous values (crime counts)
- Handles count data with Poisson-like distributions
- Can predict both mean and confidence intervals

✅ **Temporal Pattern Recognition**
- Excellent with lag features (yesterday's crimes, last week's crimes)
- Captures rolling averages and trends
- Learns seasonal patterns automatically

✅ **Spatial Awareness**
- Handles area encoding naturally
- Learns geographic crime patterns
- Captures area-specific temporal variations

✅ **Non-Linear Relationships**
- Crime patterns are non-linear (e.g., Friday night ≠ Monday morning)
- Captures complex time × area × season interactions
- No manual feature interaction needed

✅ **Fast Inference**
- Real-time predictions (<1ms per forecast)
- Suitable for operational police deployment systems
- Can forecast multiple areas/times in parallel

✅ **Feature Importance**
- Shows which factors drive crime counts
- Helps police understand "why" certain areas need more resources
- Enables strategic decision-making

**Technical Configuration:**
```python
XGBRegressor(
    objective='reg:squarederror',  # For count prediction
    n_estimators=200,              # Number of trees
    max_depth=6,                   # Tree complexity
    learning_rate=0.1,             # Conservative learning
    subsample=0.8,                 # Prevent overfitting
    colsample_bytree=0.8,          # Feature sampling
)
```

#### 4.3 Alternative Models (Comparison)

##### **Prophet (Facebook Time-Series)** ⭐⭐⭐⭐
- **Best for**: Pure time-series forecasting with seasonality
- **Pros**: Handles holidays, seasonality automatically
- **Cons**: Less flexible with spatial features, slower than XGBoost
- **Use Case**: Good for area-specific time-series predictions

##### **Random Forest Regressor** ⭐⭐⭐⭐
- **Best for**: Similar to XGBoost but more interpretable
- **Pros**: Robust, handles outliers well, easy to explain
- **Cons**: Slightly lower accuracy than XGBoost, slower training
- **Use Case**: Good alternative if explainability is critical

##### **ARIMA/SARIMA** ⭐⭐⭐
- **Best for**: Traditional time-series forecasting
- **Pros**: Well-understood, statistical foundations
- **Cons**: Requires stationary data, doesn't handle spatial features well
- **Use Case**: Baseline model for comparison

##### **LSTM Neural Networks** ⭐⭐⭐
- **Best for**: Sequential patterns with lots of data
- **Pros**: Captures long-term dependencies, very flexible
- **Cons**: Needs more data, harder to train, less interpretable
- **Use Case**: Consider if dataset is very large (>5M records)

##### **Poisson Regression** ⭐⭐
- **Best for**: Count data with statistical rigor
- **Pros**: Specifically designed for count data, interpretable coefficients
- **Cons**: Assumes linear relationships, limited non-linearity
- **Use Case**: Baseline statistical model

#### 4.4 Recommended Architecture

**Primary Model: XGBoost Regressor**
- Main forecasting engine
- Optimized for speed and accuracy

**Ensemble Enhancement (Optional):**
```
Final Prediction = 0.7 × XGBoost + 0.3 × Prophet
```
- Combines XGBoost's accuracy with Prophet's seasonality handling
- Improves robustness

**Two-Stage Approach:**
1. **Stage 1**: Predict crime COUNT (XGBoost Regressor)
2. **Stage 2**: Predict crime TYPE distribution (XGBoost Multi-class Classifier)

This gives both "how many" and "what types"

#### 4.5 Key Features for Forecasting

**Temporal Features:**
- Year, month, day, day_of_week
- Hour, time_block (3-hour windows)
- Is_weekend, is_holiday
- Week_of_year, season

**Spatial Features:**
- Area code/name (encoded)
- District number
- Geographic clusters

**Historical Features (Critical for Forecasting):**
- crime_count_lag1 (yesterday's count)
- crime_count_lag7 (last week's count)
- crime_count_rolling_7 (7-day average)
- crime_count_rolling_30 (30-day average)
- Year-over-year comparison

**Interaction Features:**
- Area × Time_block (some areas peak at different times)
- Day_of_week × Hour (weekend vs weekday patterns)
- Month × Area (seasonal variations by location)

#### 4.6 Evaluation Metrics

**For Crime Count Prediction (Regression):**
- **MAE (Mean Absolute Error)**: Average prediction error in number of crimes
  - Target: <5 crimes error for 3-hour windows
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **R² Score**: Proportion of variance explained
  - Target: >0.7 (explains 70%+ of crime variation)
- **MAPE (Mean Absolute Percentage Error)**: Relative error
  - Target: <20% error

**For Crime Type Distribution (Classification):**
- **Top-K Accuracy**: Are actual crime types in top K predictions?
- **Cross-Entropy Loss**: Quality of probability distributions
- **Precision/Recall per Crime Type**: For specialized unit deployment

#### 4.7 Model Interpretability (Critical for Police Adoption)

**SHAP Values:**
- Explain individual predictions
- Show why model predicts high/low crime for specific area+time
- Build trust with police departments

**Feature Importance:**
- Rank features by impact on predictions
- Helps police understand crime drivers
- Enables strategic interventions

**Confidence Intervals:**
- Provide prediction ranges, not just point estimates
- Help police plan for uncertainty
- Critical for resource allocation decisions



## Recommended Implementation Strategy

### Phase 1: Data Preparation & Aggregation (Weeks 1-2)
1. **Data Cleaning**: Handle missing values, validate timestamps
2. **Data Aggregation**: Transform from individual crimes to time-window aggregates
   - Group by: Area + Date + Time Block
   - Calculate: Crime counts, crime type distributions
3. **Feature Engineering**: Create lag features, rolling averages, temporal features

### Phase 2: Crime Count Forecasting Model (Weeks 3-4)
1. **XGBoost Regressor Development**: Train model to predict crime counts
2. **Hyperparameter Tuning**: Optimize model parameters using cross-validation
3. **Feature Selection**: Use feature importance to optimize feature set
4. **Model Validation**: Test on holdout time periods (future data)

### Phase 3: Crime Type Distribution Model (Weeks 5-6)
1. **Multi-Output Classifier**: Predict probability of each crime type
2. **Integration**: Combine count + type predictions into single forecast
3. **Validation**: Ensure crime type predictions align with historical patterns

### Phase 4: Prediction Interface & Deployment (Weeks 7-8)
1. **API Development**: Create prediction interface
   - Input: Area + Time Window
   - Output: Crime count + Type distribution + Confidence intervals
2. **Visualization Dashboard**: Interactive forecasting tool
3. **Performance Monitoring**: Track prediction accuracy over time

### Phase 5: Advanced Features (Weeks 9-12)
1. **Ensemble Methods**: Combine XGBoost with Prophet/ARIMA
2. **External Features**: Add weather, events, holidays
3. **Real-time Updates**: Continuous learning from new crime data
4. **Explainability Tools**: SHAP values and feature importance dashboards

## Model Selection Justification

### **Selected Approach: XGBoost Regressor for Crime Forecasting**

**Why XGBoost Regressor is OPTIMAL for Crime Count Forecasting:**

1. **Correct Problem Framing**: 
   - Crime forecasting is REGRESSION (predicting counts), not classification
   - XGBRegressor is specifically designed for count prediction
   - Handles continuous target variables with high accuracy

2. **Superior Performance for Structured Data**:
   - Consistently outperforms alternatives on tabular datasets
   - Excellent for time-series with additional features
   - Proven track record in forecasting competitions

3. **Temporal Pattern Recognition**:
   - Naturally handles lag features (historical crime counts)
   - Captures rolling averages and trends
   - Learns complex seasonal and cyclic patterns

4. **Spatial Awareness**:
   - Handles geographic features (area encoding) effectively
   - Learns area-specific crime patterns
   - Captures spatial-temporal interactions

5. **Non-Linear Relationships**:
   - Crime patterns are inherently non-linear
   - Automatically discovers feature interactions
   - No need for manual polynomial features

6. **Fast Inference**:
   - Real-time predictions (<1ms per forecast)
   - Scalable to citywide deployment
   - Suitable for operational police systems

7. **Feature Importance & Explainability**:
   - Shows which factors drive crime forecasts
   - Enables SHAP value integration
   - Critical for police department adoption and trust

8. **Robust to Missing Data**:
   - Handles missing time/location values naturally
   - No need for extensive imputation
   - Maintains accuracy with incomplete records

9. **Production-Ready**:
   - Mature libraries (xgboost, scikit-learn)
   - Easy deployment and maintenance
   - Strong community support

**Why NOT Classification:**
- Classification answers "what category is this crime?"
- We need to answer "how many crimes will occur?"
- These are fundamentally different problems requiring different models

**Comparison with Alternatives:**

| Model | Accuracy | Speed | Interpretability | Temporal Features | Spatial Features | Overall |
|-------|----------|-------|------------------|-------------------|------------------|---------|
| **XGBoost Regressor** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Best** |
| Prophet | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Good |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Good |
| ARIMA/SARIMA | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | Fair |
| LSTM | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Fair |
| Poisson Regression | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Baseline |
6. **Scalability**: Efficiently processes large LA crime datasets
7. **SHAP Integration**: Native support for explainable AI requirements

### **Why Single Model Approach:**
- **Focused Implementation**: Concentrated effort on optimizing one high-performance model
- **Resource Efficiency**: Better allocation of development and computational resources
- **Deployment Simplicity**: Easier maintenance and monitoring in production
- **Expertise Development**: Deep specialization in XGBoost optimization techniques

### 5. Data Processing Requirements
- **Data Cleaning**: Handle missing values, outliers, and inconsistent data entries
- **Feature Engineering** (Based on Available Data): 
  - Extract temporal features from Date Occurred and Time Occurred (hour, day, month, season)
  - Create geographical features from area codes, reporting districts, and coordinates
  - Generate crime type categories from crime codes and descriptions
  - Calculate victim demographic distributions when available
  - Create premise-based risk categories from premise codes
  - Develop weapon usage indicators from weapon data
- **Data Validation**: Ensure data quality and consistency for available fields

### 6. Analysis Components

#### 6.1 Exploratory Data Analysis (EDA)
- Statistical summary of crime data
- Distribution analysis by location and time
- Correlation analysis between variables
- Visualization of crime patterns and trends

#### 6.2 Predictive Analytics (Based on Available Data)
- Crime occurrence probability modeling using XGBoost
- Crime type classification based on crime codes
- Geographic hotspot prediction using area codes and coordinates
- Time-based crime forecasting from temporal fields
- Premise-specific risk assessment
- Victim demographic pattern analysis when data is available

#### 6.3 Optimization Engine
- Police patrol route optimization
- Resource allocation algorithms
- Response time optimization
- Coverage area maximization

### 7. Performance Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Regression Metrics**: MSE, RMSE, MAE, R²
- **Time Series Metrics**: MAPE, SMAPE, AIC, BIC
- **Spatial Accuracy**: Geographic prediction accuracy

## Technical Stack Requirements

### 8. Programming Languages & Libraries
- **Python**: Primary development language
- **Key Libraries**:
  - **Data Processing**: pandas, numpy, scipy
  - **Machine Learning**: XGBoost (primary), scikit-learn (preprocessing)
  - **Model Explainability**: SHAP for XGBoost interpretability
  - **Geospatial**: geopandas, folium, shapely
  - **Visualization**: matplotlib, seaborn, plotly, bokeh

### 9. Infrastructure Requirements
- **Computing Resources**: High-performance computing for model training
- **Storage**: Sufficient space for large datasets and model artifacts
- **Database**: For data storage and retrieval (PostgreSQL with PostGIS extension recommended)
- **Deployment**: Cloud platform for model serving (AWS/Azure/GCP)

## Deliverables

### 10. Expected Outputs (Based on Available Dataset)
1. **Analysis Reports**:
   - Crime pattern analysis by area codes and time periods
   - Hotspot identification using coordinates and reporting districts
   - Temporal analysis from Date/Time Occurred fields
   - Crime type distribution and trend analysis

2. **Predictive Models**:
   - Trained XGBoost model for crime occurrence prediction
   - Model performance evaluation reports
   - Feature importance analysis for available data fields

3. **Visualization Dashboard**:
   - Interactive crime maps using LAT/LON coordinates
   - Time series charts from temporal data
   - Area-based risk assessment interface
   - Crime type distribution visualizations

4. **Deployment Package**:
   - Production-ready XGBoost prediction system
   - API endpoints for real-time crime risk queries
   - Documentation and user guides

5. **Technical Report (report.md)**:
   - Comprehensive documentation following academic standards
   - Detailed analysis and methodology documentation
   - Complete project report with findings and conclusions

## Documentation Requirements (report.md)

### 11. Report Structure & Content Guidelines

The final technical report (report.md) must follow this exact structure:

#### **1. Abstract (250-500 words)**
- **Content Requirements**:
  - Clear description of the predictive policing project objectives
  - Overview of machine learning techniques applied
  - Summary of key findings and outcomes
  - Brief mention of practical implications for police deployment
- **Format**: Single paragraph, concise and comprehensive

#### **2. Introduction to the Topic**
- **Section Purpose**: Comprehensive background on predictive policing and machine learning

##### **(i) Brief History of the Topic**
- Evolution of predictive policing from traditional methods
- Timeline of machine learning adoption in law enforcement
- Key milestones and breakthrough applications

##### **(ii) What Types of Problems it Solves?**
- Crime hotspot identification and prediction
- Optimal resource allocation and patrol route planning
- Temporal crime pattern analysis
- Public safety enhancement through data-driven insights

##### **(iii) What Types of Data it Needs?**
- Historical crime incident data (locations, times, types)
- Geographic and demographic information
- Temporal data (dates, times, seasonal patterns)
- Categorical crime classification data

##### **(iv) Who is Using it?**
- Law enforcement agencies (LAPD, NYPD, Chicago PD)
- Municipal governments and public safety departments
- Research institutions and academic organizations
- Private security companies and consultancies

##### **(v) Possible Software or Packages to Use**
- **Primary Tools**: Python, R, MATLAB
- **ML Libraries**: scikit-learn, XGBoost, TensorFlow, PyTorch
- **Visualization**: matplotlib, seaborn, plotly, folium
- **Data Processing**: pandas, numpy, geopandas
- **Statistical Analysis**: statsmodels, scipy

#### **3. Demonstration of the Technique**
- **Section Purpose**: Detailed methodology and implementation

##### **(i) What is the Purpose of the Analysis?**
- Identify high-crime zones in Los Angeles
- Predict optimal timing for police deployment
- Analyze crime patterns and trends from 2020-present
- Develop actionable insights for resource allocation

##### **(ii) What Software Package are You Using?**
- Detailed justification for chosen tools and libraries
- Version specifications and compatibility requirements
- Integration approach for multiple tools

##### **(iii) How the Output Should be Interpreted?**
- **Model Performance Metrics**: Accuracy, precision, recall interpretation
- **Visualization Guidance**: Heat maps, time series plots, confusion matrices
- **Prediction Confidence**: Probability thresholds and confidence intervals
- **Business Impact**: Translation of technical results to actionable police strategies

##### **(iv) Comment on the Output**
- Critical analysis of model performance
- Identification of strengths and limitations
- Comparison between different ML techniques
- Real-world applicability assessment

##### **(v) Code Demonstration**
- **Restriction**: Only small, essential code snippets allowed
- **Examples**: Key model training code, critical preprocessing steps
- **Requirement**: All extensive code must be in Appendix

#### **4. Conclusion**

##### **(i) Personal View of Technique Usefulness (Individual Reflection)**
- Critical evaluation of predictive policing effectiveness
- Discussion of ethical considerations and potential biases
- Assessment of practical implementation challenges
- Personal insights on future development directions

##### **(ii) Summary of Report Content**
- Concise recap of methodology and findings
- Key achievements and deliverables
- Impact on police deployment strategies
- Recommendations for future work

#### **5. References**
- **Format**: Academic citation style (APA/IEEE recommended)
- **Requirements**: Minimum 15-20 credible sources
- **Content**: Research papers, official reports, technical documentation
- **Quality**: Peer-reviewed sources preferred

#### **6. Appendix**
- **Code Repository**: Complete implementation code
- **Extended Figures**: Additional visualizations and charts
- **Data Samples**: Representative data examples (NO FULL DATASET)
- **Detailed Results**: Comprehensive model outputs and metrics
- **Technical Specifications**: Hardware/software configuration details

### 12. Report Formatting Standards

#### **Critical Formatting Requirements:**

##### **Figure Management**
- **Caption Requirement**: Every figure MUST have descriptive captions
- **Example Format**: "Figure 1: Crime hotspot distribution across LA neighborhoods showing highest concentration in downtown areas"
- **Text Reference**: All figures must be referenced in text (e.g., "As shown in Figure 1...")
- **Placement**: Figures should be placed near relevant text

##### **Data Inclusion Policy**
- **PROHIBITED**: Full datasets in main report or appendix
- **ALLOWED**: Small representative samples in appendix only
- **DEMONSTRATION**: Statistical summaries and sample records only

##### **Code Inclusion Guidelines**
- **Main Report**: Maximum 10-15 lines of critical code per section
- **Purpose**: Demonstrate key concepts only
- **Appendix**: Complete code with proper documentation
- **Format**: Use code blocks with syntax highlighting

##### **Page Numbering**
- **Requirement**: MANDATORY page numbers on every page
- **Format**: Bottom center or top right corner
- **Style**: "Page X of Y" format recommended

##### **Output Presentation**
- **Focus**: Only important results with detailed commentary
- **Avoid**: Extensive raw output without analysis
- **Emphasis**: Your interpretation and insights are crucial
- **Balance**: Equal weight to results and analysis

##### **Report Length Guidelines**
- **Target**: Quality over quantity - concise and focused
- **Sections**: Balanced content across all required sections
- **Commentary**: Extensive analysis and interpretation required
- **Redundancy**: Avoid repetitive content

### 13. Quality Assurance Checklist
- [ ] Abstract within 250-500 word limit
- [ ] All required sections included and complete
- [ ] Figures numbered and captioned properly
- [ ] No raw datasets included
- [ ] Page numbers on every page
- [ ] Code snippets minimal in main report
- [ ] Comprehensive appendix with complete code
- [ ] References properly formatted
- [ ] Personal reflection included in conclusion
- [ ] Output interpretation and commentary provided

## Success Criteria

### 11. Measurable Outcomes (Based on Available Dataset)
- **Accuracy**: Achieve >80% accuracy in crime type prediction using available crime codes
- **Geographic Precision**: Predict high-crime areas using area codes and coordinates with >75% accuracy
- **Temporal Analysis**: Identify peak crime hours from TIME OCC data with >70% precision
- **Resource Optimization**: Improve patrol deployment efficiency using area-based predictions
- **Feature Insights**: Provide clear importance rankings for temporal, geographic, and crime type features

## Timeline & Milestones
1. **Week 1-2**: Data exploration and preprocessing
2. **Week 3-4**: Feature engineering and EDA
3. **Week 5-7**: Model development and training
4. **Week 8-9**: Model evaluation and optimization
5. **Week 10-11**: Dashboard development and testing
6. **Week 12**: Deployment and documentation

## Risk Factors & Mitigation
- **Data Quality Issues**: Implement robust data validation procedures
- **Model Overfitting**: Use cross-validation and regularization techniques
- **Computational Complexity**: Leverage cloud computing and optimization algorithms
- **Real-time Performance**: Design efficient algorithms and caching strategies

## Ethical Considerations
- Ensure unbiased model development
- Protect citizen privacy and data security
- Avoid discriminatory policing patterns
- Maintain transparency in algorithmic decision-making

---

*This requirements document serves as the foundation for developing an effective predictive policing system that enhances public safety through data-driven insights and optimal resource allocation.*