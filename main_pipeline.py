from data_loader import load_story_points_dataset, preprocess_data
from deepseek_estimator import DeepSeekEstimator
from effort_calculator import EffortCalculator
import pandas as pd

def run_estimation_pipeline():
    """Main pipeline: Load 100 stories -> Estimate with DeepSeek -> Calculate effort"""
    
    # 1. Load 100 sample user stories from CSV
    print("Loading 100 user stories from CSV...")
    df = load_story_points_dataset()
    df = preprocess_data(df)
    print(f"Loaded {len(df)} stories")
    
    # 2. Estimate story points using DeepSeek
    print("Estimating story points using DeepSeek...")
    estimator = DeepSeekEstimator()
    story_points = estimator.estimate_batch(df)
    df['estimated_story_points'] = story_points
    
    # 3. Calculate effort estimation
    print("Calculating effort estimation...")
    calculator = EffortCalculator()
    summary = calculator.generate_summary(df, story_points)
    
    # Results
    print("\n=== ESTIMATION RESULTS ===")
    effort = summary['effort_estimation']
    print(f"Total Story Points: {effort['total_story_points']}")
    print(f"Estimated Hours: {effort['estimated_hours']}")
    print(f"Estimated Days: {effort['estimated_days']:.1f}")
    print(f"Estimated Sprints: {effort['estimated_sprints']:.0f}")
    print(f"Team Weeks: {effort['team_weeks']:.0f}")
    
    return {'dataframe': df, 'summary': summary}

if __name__ == "__main__":
    results = run_estimation_pipeline()