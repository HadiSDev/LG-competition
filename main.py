import os

from runs.A2C import run_A2C
import pandas as pd

if __name__ == '__main__':
    # Test Permissions
    CSV = {
        "Levels": [],
        "Rewards": [],
        "total_steps": [],
        "new_progress": [],
        "successful_clicks": [],
        "goal_reached": [],
        "valid_steps": []

    }

    df = pd.DataFrame(CSV)
    os.makedirs("logs")
    df.to_csv(f"./logs/TEST_PERM")


    run_A2C()
