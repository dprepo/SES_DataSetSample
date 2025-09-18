import openai
import numpy as np
import pandas as pd

class DeepSeekEstimator:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=""  # Replace with actual API key
        )
    
    def estimate_story_points(self, description, complexity="Medium", priority="Medium"):
        """Estimate story points using DeepSeek"""
        prompt = f"""
Estimate story points for this user story using Fibonacci sequence (1,2,3,5,8,13,21):

Story: {description}
Complexity: {complexity}
Priority: {priority}

Consider:
- Low complexity: 1-3 points
- Medium complexity: 3-8 points  
- High complexity: 8-21 points

Respond with only the numeric value.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek/deepseek-r1:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            points = int(response.choices[0].message.content.strip())
            return points if points in [1,2,3,5,8,13,21] else 5
        except:
            # Fallback based on complexity
            complexity_map = {"Low": 2, "Medium": 5, "High": 13}
            return complexity_map.get(complexity, 5)
    
    def estimate_batch(self, df):
        """Estimate story points for all stories"""
        story_points = []
        for _, row in df.iterrows():
            points = self.estimate_story_points(
                row['description'], 
                row.get('complexity', 'Medium'),
                row.get('priority', 'Medium')
            )
            story_points.append(points)
        return story_points