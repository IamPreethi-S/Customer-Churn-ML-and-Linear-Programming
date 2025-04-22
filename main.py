from src.preprocessing import preprocess_data
from src.modeling import train_model
from src.optimization import optimize_retention
from src.visualization import plot_results

if __name__ == "__main__":
    print("Running preprocessing...")
    preprocess_data()
    
    print("\nTraining model...")
    train_model()
    
    print("\nOptimizing retention...")
    optimize_retention()
    
    print("\nGenerating visualizations...")
    plot_results()
    
    print("\nPipeline completed! Results saved to /output")