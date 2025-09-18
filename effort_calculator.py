import pandas as pd
import numpy as np

class EffortCalculator:
    def __init__(self, hours_per_point=8, team_velocity=20):
        self.hours_per_point = hours_per_point
        self.team_velocity = team_velocity
    
    def calculate_effort(self, story_points):
        """Calculate effort metrics from story points"""
        total_points = sum(story_points)
        
        return {
            'total_story_points': total_points,
            'estimated_hours': total_points * self.hours_per_point,
            'estimated_days': (total_points * self.hours_per_point) / 8,
            'estimated_sprints': np.ceil(total_points / self.team_velocity),
            'team_weeks': np.ceil(total_points / self.team_velocity) * 2
        }
    
    def generate_summary(self, df, story_points):
        """Generate effort estimation summary"""
        effort = self.calculate_effort(story_points)
        
        # Priority breakdown
        priority_breakdown = df.groupby('priority').size().to_dict()
        complexity_breakdown = df.groupby('complexity').size().to_dict()
        
        return {
            'effort_estimation': effort,
            'priority_breakdown': priority_breakdown,
            'complexity_breakdown': complexity_breakdown,
            'average_points_per_story': np.mean(story_points),
            'point_distribution': pd.Series(story_points).value_counts().to_dict()
        }