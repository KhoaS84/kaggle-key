"""
Local Simulator script for testing agents in Kaggriculture.
"""
from kaggle_environments import make

def main():
    print("Starting Kaggriculture local simulation (720 turns = 30 days)...")
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    
    # Run our agent against the built-in starter or random agent
    env.run(["main.py", "starter"])
    
    final_step = env.steps[-1]
    print("=== Simulation Complete ===")
    for i, s in enumerate(final_step):
        print(f"Player {i}: final money/reward = {s.reward}, status = {s.status}")

if __name__ == "__main__":
    main()
