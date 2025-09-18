import openai
import json
import numpy as np
from typing import List, Dict

class GEvalFramework:
    def __init__(self, model_name="deepseek/deepseek-r1:free"):
        self.model_name = model_name
        self.client = openai.OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=""  # Replace with actual API key
        )
    
    def create_evaluation_prompt(self, story_description: str, predicted_points: int, actual_points: int) -> str:
        """Create G-Eval prompt for story point estimation"""
        return f"""
You are an expert software project manager evaluating story point estimations.

Task: Evaluate the accuracy of a story point estimation for the following user story.

User Story: {story_description}
Predicted Story Points: {predicted_points}
Actual Story Points: {actual_points}

Evaluation Criteria:
1. Accuracy: How close is the prediction to the actual value?
2. Reasonableness: Does the estimation make sense given the story complexity?

Please provide a score from 1-5 where:
1 = Very poor estimation (>50% error)
2 = Poor estimation (25-50% error)
3 = Fair estimation (10-25% error)
4 = Good estimation (5-10% error)
5 = Excellent estimation (<5% error)

Respond with only the numeric score (1-5).
"""
    
    def evaluate_single_prediction(self, story_description: str, predicted_points: int, actual_points: int) -> float:
        """Evaluate a single story point prediction using G-Eval"""
        prompt = self.create_evaluation_prompt(story_description, predicted_points, actual_points)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            score = float(response.choices[0].message.content.strip())
            return max(1, min(5, score))  # Ensure score is between 1-5
        except:
            # Fallback scoring based on percentage error
            error_pct = abs(predicted_points - actual_points) / actual_points * 100
            if error_pct < 5: return 5
            elif error_pct < 10: return 4
            elif error_pct < 25: return 3
            elif error_pct < 50: return 2
            else: return 1
    
    def evaluate_batch(self, stories: List[str], predictions: List[int], actuals: List[int]) -> Dict:
        """Evaluate batch of predictions"""
        scores = []
        for story, pred, actual in zip(stories, predictions, actuals):
            score = self.evaluate_single_prediction(story, pred, actual)
            scores.append(score)
        
        return {
            'individual_scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }