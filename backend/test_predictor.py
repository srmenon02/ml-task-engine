import structlog
from core.predictor import get_predictor
from core.accuracy_tracker import calculate_prediction_accuracy

logger = structlog.get_logger()

def main():
    print("Resource Predictor Test")
    predictor = get_predictor()

    print(f"\nPredictor Trained: {predictor.is_trained}")
    print(f"Training Samples: {predictor.training_samples}")

    if not predictor.is_trained:
        print("\nPredictor not trained. Training predictor now")
        success = predictor.train(min_samples=5)
        if success:
            print("Training successful")
        else:
            print("Training failed, need at least 5 completed jobs")
            return
        
    print("Test Predictions\n")
    test_cases = [
        {"n_estimators": 10, "dataset_rows": 1000},
        {"n_estimators": 50, "dataset_rows": 10000},
        {"n_estimators": 200, "dataset_rows": 50000}
    ]

    for case in test_cases:
        memory, cpu = predictor.predict(case, "train_sklearn_model")
        print(f"\nConfig: {case}")
        print(f"    Predicted Memory: {memory:.1f} MB")
        print(f"    Predicted CPU: {cpu:.1f}%")

    print("Accuracy Metrics\n")

    metrics = predictor.evaluate()

    if "error" in metrics:
        print(f"error: {metrics['error']}")
    else:
        print(f"    Samples: {metrics['samples']}")
        print(f"    Memory MAE: {metrics['memory_mae_mb']:.1f}")
        print(f"    Memory MAPE: {metrics['memory_mape_percent']:.1f}")
        print(f"    CPU MAE: {metrics['cpu_mae_percent']:.1f}")
        print(f"    CPU MAPE: {metrics['cpu_mape_percent']:.1f}")

    print("Predicted v. Actual")

    accuracy = calculate_prediction_accuracy()

    if "error" in accuracy:
        print(f"Error: {accuracy['error']}")
    else:
        print(f"    Total Jobs Analyzed: {accuracy['total jobs']}")
        if accuracy['memory_mape']:
            print(f"    Memory MAPE: {accuracy['memory_mape']:.1f}%")
            print(f"    Memory predictions < 20%: {accuracy['memory_predictions_within_20_percent']:.1f}%")
        if accuracy['cpu_mape']:
            print(f"    CPU MAPE: {accuracy['cpu_mape']:.1f}%")
            print(f"    CPU predictions < 20%: {accuracy['cpu_predictions_within_20_percent']:.1f}%")

if __name__ == "__main__":
    main()

