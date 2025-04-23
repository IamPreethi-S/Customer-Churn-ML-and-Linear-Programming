from src.preprocessing import preprocess_data
from src.modeling import train_model
from src.optimization import optimize_retention
from src.visualization import plot_results
from src.budget_simulation import simulate_budget_curve

if __name__ == "__main__":
    print("Preprocessing the data...")
    preprocess_data()

    print("\nModel Training...")
    train_model()
    print("\nModel Training completed...")
    
    print("\nOptimizing retention...")
    optimize_retention()
    
    print("\nGenerating visualizations...")
    plot_results()
    
    print("\nResults saved to output directory")

