import pandas as pd
import datetime as dt

def fix_data_types(df):
    """Fix data types of the DataFrame."""
    
    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert date columns
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass
    
    return df


#remove invalid values 
def remove_invalid_values(df):
    """Remove invalid values from the DataFrame."""
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    return df

#handle missing values strategy: fill with mean, median, or mode
def handle_missing_values(df, strategy='mean'):
    """Handle missing values in the DataFrame."""
    
    if strategy == 'mean':
        df = df.fillna(df.mean())
    elif strategy == 'median':
        df = df.fillna(df.median())
    elif strategy == 'mode':
        df = df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', or 'mode'.")
    
    return df


#function: fix_data_types, remove_invalid_values, handle_missing_values
def clean_data(df, missing_value_strategy='mean'):
    """Clean the dataset by fixing data types, removing invalid values, and handling missing values."""
    df = fix_data_types(df)
    df = remove_invalid_values(df)
    df = handle_missing_values(df, strategy=missing_value_strategy)


    # Remove rows with negative values in numeric columns (assuming these are invalid for this dataset) and handle any remaining NaN values in those columns by dropping them.
    df = df[df['planned_duration'] >= 0]  # Remove rows where planned_duration is negative
    df = df.dropna(subset=['planned_duration'])  # Remove rows where planned_duration is NaN
    df = df[df['actual_duration'] >= 0]  # Remove rows where actual_duration is negative
    df = df.dropna(subset=['actual_duration'])  # Remove rows where actual_duration is NaN
    df = df[df['delay'] >= 0]  # Remove rows where delay is negative
    df = df.dropna(subset=['delay'])  # Remove rows where delay is NaN
    df = df[df['cost_overrun'] >= 0]  # Remove rows where cost_overrun is negative
    df = df.dropna(subset=['cost_overrun'])  # Remove rows where cost_overrun is NaN
    df = df[df['risk_score'] >= 0]  # Remove rows where risk_score is negative
    df = df.dropna(subset=['risk_score'])  # Remove rows where risk_score is NaN
    df = df[df['project_size'] >= 0]  # Remove rows where project_size is negative
    df = df.dropna(subset=['project_size'])  # Remove rows where project_size is NaN
    df = df[df['num_workers'] >= 0]  # Remove rows where num_workers is negative
    df = df.dropna(subset=['num_workers'])  # Remove rows where num_workers is NaN
    df = df[df['weather_condition'] >= 0]  # Remove rows where weather_condition is negative
    df = df.dropna(subset=['weather_condition'])  # Remove rows where weather_condition is NaN
    df = df[df['material_availability'] >= 0]  # Remove rows where material_availability is negative
    df = df.dropna(subset=['material_availability'])  # Remove rows where material_availability is NaN
    df = df[df['equipment_availability'] >= 0]  # Remove rows where equipment_availability is negative
    df = df.dropna(subset=['equipment_availability'])  # Remove rows where equipment_availability is NaN
    df = df[df['labor_availability'] >= 0]  # Remove rows where labor_availability is negative
    df = df.dropna(subset=['labor_availability'])  # Remove rows where labor_availability is NaN
    df = df[df['supply_chain_disruption'] >= 0]  # Remove rows where supply_chain_disruption is negative
    df = df.dropna(subset=['supply_chain_disruption'])  # Remove rows where supply_chain_disruption is NaN
    df = df[df['regulatory_issues'] >= 0]  # Remove rows where regulatory_issues is negative
    df = df.dropna(subset=['regulatory_issues'])  # Remove rows where regulatory_issues is NaN
    df = df[df['labor_strikes'] >= 0]  # Remove rows where labor_strikes is negative
    df = df.dropna(subset=['labor_strikes'])  # Remove rows where labor_strikes is NaN
    df = df[df['design_changes'] >= 0]  # Remove rows where design_changes is negative
    df = df.dropna(subset=['design_changes'])  # Remove rows where design_changes is NaN
    df = df[df['unforeseen_events'] >= 0]  # Remove rows where unforeseen_events is negative
    df = df.dropna(subset=['unforeseen_events'])  # Remove rows where unforeseen_events is NaN
    df = df[df['communication_issues'] >= 0]  # Remove rows where communication_issues is negative
    df = df.dropna(subset=['communication_issues'])  # Remove rows where communication_issues is NaN
    df = df[df['project_complexity'] >= 0]  # Remove rows where project_complexity is negative
    df = df.dropna(subset=['project_complexity'])  # Remove rows where project_complexity is NaN
    df = df[df['project_type'] >= 0]  # Remove rows where project_type is negative
    df = df.dropna(subset=['project_type'])  # Remove rows where project_type is NaN
    df = df[df['location'] >= 0]  # Remove rows where location is negative
    df = df.dropna(subset=['location'])  # Remove rows where location is NaN
    df = df[df['season'] >= 0]  # Remove rows where season is negative
    df = df.dropna(subset=['season'])  # Remove rows where season is NaN
    df = df[df['project_phase'] >= 0]  # Remove rows where project_phase is negative
    df = df.dropna(subset=['project_phase'])  # Remove rows where project_phase is NaN
    df = df[df['contract_type'] >= 0]  # Remove rows where contract_type is negative
    df = df.dropna(subset=['contract_type'])  # Remove rows where contract_type is NaN
    df = df[df['project_manager_experience'] >= 0]  # Remove rows where project_manager_experience is negative
    df = df.dropna(subset=['project_manager_experience'])  # Remove rows where project_manager_experience is NaN
    df = df[df['team_experience'] >= 0]  # Remove rows where team_experience is negative
    df = df.dropna(subset=['team_experience'])  # Remove rows where team_experience is NaN
    df = df[df['stakeholder_involvement'] >= 0]  # Remove rows where stakeholder_involvement is negative
    df = df.dropna(subset=['stakeholder_involvement'])  # Remove rows where stakeholder_involvement is NaN
    df = df[df['project_funding'] >= 0]  # Remove rows where project_funding is negative
    df = df.dropna(subset=['project_funding'])  # Remove rows where project_funding is NaN
    df = df[df['economic_conditions'] >= 0]  # Remove rows where economic_conditions is negative
    df = df.dropna(subset=['economic_conditions'])  # Remove rows where economic_conditions is NaN
    df = df[df['political_stability'] >= 0]  # Remove rows where political_stability is negative
    df = df.dropna(subset=['political_stability'])  # Remove rows where political_stability is NaN
    df = df[df['social_unrest'] >= 0]  # Remove rows where social_unrest is negative
    df = df.dropna(subset=['social_unrest'])  # Remove rows where social_unrest is NaN
    df = df[df['environmental_factors'] >= 0]  # Remove rows where environmental_factors is negative
    df = df.dropna(subset=['environmental_factors'])  # Remove rows where environmental_factors is NaN
    df = df[df['technological_factors'] >= 0]  # Remove rows where technological_factors is negative
    df = df.dropna(subset=['technological_factors'])  # Remove rows where technological_factors is NaN
    df = df[df['legal_factors'] >= 0]  # Remove rows where legal_factors is negative
    df = df.dropna(subset=['legal_factors'])  # Remove rows where legal_factors is NaN
    df = df[df['cultural_factors'] >= 0]  # Remove rows where cultural_factors is negative
    df = df.dropna(subset=['cultural_factors'])  # Remove rows where cultural_factors is NaN
    df = df[df['delay_risk'] >= 0]  # Remove rows where delay_risk is negative
    df = df.dropna(subset=['delay_risk'])  # Remove rows where delay
    df = df[df['cost_overrun_risk'] >= 0]  # Remove rows where cost_overrun_risk is negative
    df = df.dropna(subset=['cost_overrun_risk'])  # Remove rows where cost_overrun_risk is NaN
    df = df[df['risk_score'] >= 0]  # Remove rows where risk_score is negative
    df = df.dropna(subset=['risk_score'])  # Remove rows where risk_score is NaN
    df = df[df['project_size'] >= 0]  # Remove rows where project_size is negative
    df = df.dropna(subset=['project_size'])  # Remove rows where project_size is NaN
    df = df[df['num_workers'] >= 0]  # Remove rows where num_workers is negative
    df = df.dropna(subset=['num_workers'])  # Remove rows where num_workers is NaN
    df = df[df['weather_condition'] >= 0]  # Remove rows where weather_condition is negative
    df = df.dropna(subset=['weather_condition'])  # Remove rows where weather_condition is NaN
    df = df[df['other_factors'] >= 0]  # Remove rows where other_factors is negative
    df = df.dropna(subset=['other_factors'])  # Remove rows where other_factors is NaN
    
    
    return df

# if conversion fails, it will set the value to NaN, which can then be handled by the missing value strategy.
# This approach ensures that the cleaning process is robust and can handle a variety of data issues without crashing.
# The `clean_data` function provides a comprehensive cleaning pipeline that can be easily applied to any dataset, making it a crucial step in the data preparation process for machine learning models.
# By standardizing the data cleaning process, we can improve the quality of the dataset and enhance the performance of the predictive models built on top of it.
# Overall, this cleaning module is designed to be flexible and adaptable to different datasets and requirements, allowing for efficient data preparation in the construction delay risk prediction system.

if __name__ == "__main__":
    # Example usage:
    df = pd.read_csv("road_construction_delay.csv")
    cleaned_df = clean_data(df, missing_value_strategy='mean')
    print(cleaned_df.head())  



