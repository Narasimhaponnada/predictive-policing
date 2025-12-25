# Crime Prediction Analysis - Los Angeles Dataset & Algorithm Techniques

## Project Overview
This project develops a **predictive policing system** for Los Angeles using **XGBoost gradient boosting** to optimize police force deployment and enhance public safety through data-driven crime pattern analysis.

## Dataset: Los Angeles Crime Data (2020-Present)

**Source**: `Crime_Data_from_2020_to_Present.csv`  
**Official Data Source**: https://catalog.data.gov/dataset/crime-data-from-2020-to-present  
**Coverage**: Comprehensive Los Angeles crime incidents from 2020 to current date  
**Purpose**: Crime pattern analysis, hotspot identification, and temporal crime prediction  

**Available Data Features**:
- **Temporal Information**: Date Reported (`Date Rptd`), Date Occurred (`DATE OCC`), Time Occurred (`TIME OCC`) for incident patterns
- **Geographic Data**: Police area codes (`AREA`, `AREA NAME`), reporting districts (`Rpt Dist No`), coordinates (`LAT`, `LON`), location descriptions (`LOCATION`, `Cross Street`)  
- **Crime Classifications**: Crime codes (`Crm Cd`, `Crm Cd Desc`), Part 1-2 classification, multiple crime codes (`Crm Cd 1-4`)
- **Incident Details**: DR numbers (`DR_NO`), MO codes (`Mocodes`), premise information (`Premis Cd`, `Premis Desc`)
- **Victim Demographics**: Age (`Vict Age`), Sex (`Vict Sex`), Descent (`Vict Descent`)
- **Weapon Information**: Weapon codes (`Weapon Used Cd`, `Weapon Desc`) when applicable
- **Case Status**: Investigation status (`Status`, `Status Desc`)

## Machine Learning Algorithm Strategy

### 🎯 **Primary Model** (Production Deployment)

#### **Gradient Boosted Trees (XGBoost)**
- **Primary Use**: High-accuracy crime occurrence and risk-level prediction
- **Strengths**: Superior performance on structured data, handles complex feature interactions, excellent predictive accuracy, robust to overfitting
- **Applications**: Real-time crime risk assessment, optimal patrol route generation, production deployment
- **Key Features**: Built-in regularization, handles missing values naturally, provides comprehensive feature importance rankings
- **Technical Advantages**: Optimized gradient boosting implementation with parallel processing and advanced tree pruning
- **Implementation**: XGBoost classifier/regressor for comprehensive crime pattern modeling and prediction

## Model Interpretability & Explainability

### **SHAP (SHapley Additive exPlanations)**
- **Critical Requirement**: Essential for police department adoption and trust
- **Purpose**: Explains XGBoost individual predictions and global model behavior
- **XGBoost Integration**: Native support for tree-based SHAP explanations
- **Output**: Feature importance rankings, decision rationale for XGBoost deployments

### **Feature Attribution Analysis** (Based on Available Data)
- **Geographic Features**: Area codes, reporting districts, coordinate-based zones, premise types
- **Temporal Features**: Time-of-day patterns, day-of-week trends, seasonal analysis from date fields
- **Crime Type Features**: Crime code categories, Part 1-2 classifications, weapon usage patterns
- **Victim Demographics**: Age groups, demographic patterns when available
- **Location Context**: Premise types (residential, commercial, street), cross-street relationships

## Technology Implementation Stack

**Core Framework**: Python with scikit-learn ecosystem  
**Gradient Boosting**: XGBoost for optimized gradient boosting implementation  
**Data Processing**: pandas, numpy for data manipulation and analysis  
**Visualization**: matplotlib, seaborn, plotly for interactive crime dashboards  
**Geospatial**: geopandas, folium for mapping and spatial analysis  
**Model Persistence**: joblib, pickle for production model deployment  

## Expected Outcomes (Based on Available Data)

1. **Crime Hotspot Maps**: Visual identification of high-risk geographic zones using area codes and coordinates
2. **Temporal Pattern Analysis**: Peak crime hours from `TIME OCC`, day-of-week patterns from `DATE OCC`
3. **Crime Type Predictions**: Likelihood assessment for different crime codes and classifications
4. **Geographic Risk Scoring**: Area-based and coordinate-based crime likelihood assessment
5. **Premise-Specific Insights**: Crime patterns by location type (residential, commercial, street)
6. **Resource Optimization**: Data-driven patrol deployment based on area codes and reporting districts

## Success Metrics
- **Prediction Accuracy**: >85% accuracy target for crime occurrence prediction
- **Model Speed**: Fast training and inference time for real-time deployment
- **Geographic Precision**: Accurate hotspot identification within 500m radius
- **Temporal Accuracy**: Correct peak time prediction within 2-hour windows
- **Feature Interpretability**: Clear ranking and explanation of crime prediction factors