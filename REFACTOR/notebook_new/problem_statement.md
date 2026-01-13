Problem Statement                                                                                                  
                                                                                                                     
  The core problem this ML workflow tackles is crime forecasting for Los Angeles. The goal is to build a predictive  
  system that can answer two questions: how many crimes will occur in a specific area during a given time window, and
   what types of crimes are those likely to be. The practical application here is police resource allocation - if you
   know that Central LA typically sees more vehicle thefts on Friday evenings while Hollywood has higher assault     
  rates on weekends, you can deploy patrol units accordingly.                                                        
                                                                                                                     
  The dataset used is the LAPD's Crime Data from 2020 to Present, containing about 1 million crime records with 28   
  attributes including location (area name, coordinates), time of occurrence, crime type descriptions, victim        
  demographics, and weapon information.                                                                              
                                                                                                                     
  ---                                                                                                                
  ML Workflow Overview                                                                                               
                                                                                                                     
  Notebook 1: Exploratory Data Analysis                                                                              
                                                                                                                     
  The EDA notebook does the groundwork before any modelling happens.                                                 
                                                                                                                     
  Data Quality Assessment                                                                                            
                                                                                                                     
  The first thing examined was missing values. The critical columns for forecasting - date, time, area, and crime    
  type - were checked individually. TIME OCC had the most interesting issue: a large number of records were stamped  
  with exactly 12:00 PM, which turned out to be placeholder values rather than actual occurrence times. This         
  artificial spike at noon would mess up any temporal pattern analysis.                                              
                                                                                                                     
  Geographic coverage was also validated. The data spans 21 LAPD areas with varying crime frequencies, Central having
   the highest volume.                                                                                               
                                                                                                                     
  Temporal Pattern Discovery                                                                                         
                                                                                                                     
  The analysis looked at when crimes happen. There are clear hourly patterns - crimes peak in the afternoon and      
  evening hours, dip in early morning. Weekly patterns showed that weekends have slightly different distributions    
  than weekdays. Monthly and seasonal trends were examined but showed less dramatic variation.                       
                                                                                                                     
  What stood out was that different crime types have different temporal signatures. Vehicle theft peaks at night.    
  Burglary is more common during daytime hours when people are at work. Identity theft happens uniformly throughout  
  the day since it does not require physical presence.                                                               
                                                                                                                     
  Spatial Analysis                                                                                                   
                                                                                                                     
  The 21 police areas show substantial variation in crime volume. The top 5 areas account for a disproportionate     
  share of total crimes. More importantly, different areas have different crime type compositions - some are more    
  property crime heavy, others see more violent offences.                                                            
                                                                                                                     
  The area-time interaction patterns were explored too. Entertainment districts like Hollywood show stronger weekend 
  spikes compared to residential areas.                                                                              
                                                                                                                     
  EDA Conclusions                                                                                                    
                                                                                                                     
  The notebook concluded that the data is ready for modelling with a few preprocessing steps needed. It recommended  
  XGBoost as the primary algorithm, a two-stage prediction approach, and laid out specific feature engineering       
  requirements including lag features and rolling averages.                                                          
                                                                                                                     
  ---                                                                                                                
  Notebook 2: Crime Forecasting Model                                                                                
                                                                                                                     
  This notebook implements the actual prediction system.                                                             
                                                                                                                     
  Data Preprocessing                                                                                                 
                                                                                                                     
  The 12:00 PM placeholder problem was addressed through smart imputation. Rather than dropping these records or     
  leaving the spike, the notebook redistributed these timestamps using the natural hourly distribution from          
  non-placeholder times. This created a smooth and realistic hourly crime curve.                                     
                                                                                                                     
  Date and time features were extracted: year, month, day of week, hour. Crimes were also bucketed into 3-hour time  
  blocks for aggregation.                                                                                            
                                                                                                                     
  Crime types were simplified from the raw descriptions into a manageable set - Top 20 most common types plus an     
  "Others" bucket.                                                                                                   
                                                                                                                     
  Data Aggregation                                                                                                   
                                                                                                                     
  The individual crime records (about 900k rows) were transformed into an aggregated format suitable for time series 
  forecasting. Each row in the new dataset represents one area-date-time block combination, with a count of how many 
  crimes occurred and what types they were.                                                                          
                                                                                                                     
  This aggregation step is what makes the problem tractable. Instead of predicting individual crime events, the model
   predicts crime counts per time window.                                                                            
                                                                                                                     
  Feature Engineering                                                                                                
                                                                                                                     
  Lag features were created to capture recent history:                                                               
  - crime_count_lag1: how many crimes occurred in the same area, same time block, yesterday                          
  - crime_count_lag7: same thing but one week ago                                                                    
                                                                                                                     
  Rolling averages were added:                                                                                       
  - 7-day rolling mean: short term trend                                                                             
  - 30-day rolling mean: longer term trend                                                                           
                                                                                                                     
  An imputation tracking flag was also kept as a feature, allowing the model to learn any patterns associated with   
  data quality variations.                                                                                           
                                                                                                                     
  The final feature set included: area, month, day of week, hour, time block, weekend indicator, lag features,       
  rolling averages, and imputation rate.                                                                             
                                                                                                                     
  Two-Stage Prediction Approach                                                                                      
                                                                                                                     
  This is the architectural choice that defines the workflow.                                                        
                                                                                                                     
  Stage 1 uses XGBoost Regression to predict the total crime count for a given area and time window. The model was   
  configured with 100 trees, max depth of 6, learning rate of 0.1, and early stopping with 10 rounds patience.       
                                                                                                                     
  Stage 2 distributes that total count across the 21 crime type categories. This is done using historical proportions
   - if in the past, 15% of crimes in Central LA during evening hours were vehicle thefts, then 15% of the Stage 1   
  predicted total gets allocated to vehicle theft.                                                                   
                                                                                                                     
  Why this two-stage approach? It ensures mathematical consistency. The individual crime type predictions always sum 
  exactly to the total. A single model predicting 21 separate outputs would not guarantee this property. It also     
  makes the modelling more robust since Stage 1 learns from aggregate patterns which are more stable than individual 
  category trends.                                                                                                   
                                                                                                                     
  Model Evaluation                                                                                                   
                                                                                                                     
  The data was split chronologically - 80% for training on historical patterns, 20% for testing on unseen future     
  data.                                                                                                              
                                                                                                                     
  Stage 1 performance metrics:                                                                                       
  - MAE: 1.15 crimes on test set                                                                                     
  - RMSE: 1.56 crimes                                                                                                
  - R squared: 0.48                                                                                                  
  - 55% of predictions within plus or minus 1 crime                                                                  
  - 83% within plus or minus 2 crimes                                                                                
                                                                                                                     
  For a regression problem predicting counts that typically range from 0 to about 15 per time window, these numbers  
  are reasonable. The model explains about half the variance in crime counts.                                        
                                                                                                                     
  Stage 2 evaluation showed an average MAE of 0.23 crimes per crime type category. Since Stage 2 is distributing     
  totals across 21 buckets, small errors per category are expected.                                                  
                                                                                                                     
  Feature Importance                                                                                                 
                                                                                                                     
  The lag features turned out to be strong predictors, which makes intuitive sense. Yesterday's crime count in an    
  area is informative about today's count. The 7-day lag captures weekly seasonality. Area and time block features   
  also ranked high.                                                                                                  
                                                                                                                     
  ---                                                                                                                
  Summary                                                                                                            
                                                                                                                     
  This is a two-notebook ML workflow for crime forecasting in Los Angeles. The first notebook does thorough          
  exploratory analysis to understand data quality issues, temporal and spatial patterns, and crime type              
  distributions. The second notebook implements a two-stage XGBoost-based prediction system that first predicts total
   crime counts and then distributes those across specific crime categories. The practical goal is enabling smarter  
  police resource allocation by knowing where and when different types of crimes are most likely to occur.           
                                                                                                                     
  The approach is methodical: clean the data issues like placeholder timestamps, aggregate individual records into   
  forecastable time windows, engineer features that capture recent trends and patterns, train a gradient boosting    
  model, and evaluate using appropriate regression metrics.                                                          
                                                                                                                     
⏺ That covers the main aspects. The workflow is structured well - doing EDA properly   
  before jumping into modelling, addressing data quality issues systematically, and using a sensible two-stage       
  architecture that keeps the predictions internally consistent.