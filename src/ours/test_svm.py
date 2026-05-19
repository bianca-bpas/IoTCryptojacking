import sys
import logging
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from ours.dataset_scenarios.imbalanced import dataset
from ours.transform import preprocessing

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    scenario_name = "laptop"
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]

    logging.info(f"Loading scenario: {scenario_name}...")
    scenario_obj = dataset.Scenario(scenario_name) #type:ignore[reportArgumentType]

    logging.info("Running preprocessing pipeline (feature extraction & time series split)...")
    x_train, y_train, x_test, y_test = preprocessing.classical_pipeline(scenario_obj)

    logging.info(f"Extracted {x_train.shape[0]} training windows and {x_test.shape[0]} testing windows.")
    logging.info(f"Feature dimension: {x_train.shape[1]}")

    logging.info("Training SVM model...")
    model = SVC()
    model.fit(x_train, y_train)

    logging.info("Evaluating model...")
    y_pred = model.predict(x_test)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))
    print("="*50)

if __name__ == "__main__":
    main()
