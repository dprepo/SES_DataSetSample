# Story Point Estimation with DeepSeek

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your DeepSeek API key in `deepseek_estimator.py`
3. Run: `python main_pipeline.py`

## Pipeline
1. **Load 100 user stories** from CSV with complexity/priority
2. **Estimate story points** using DeepSeek model (Fibonacci: 1,2,3,5,8,13,21)
3. **Calculate effort estimation** - hours, days, sprints, team weeks

## Output
- Total story points
- Estimated hours/days
- Sprint count
- Team weeks required