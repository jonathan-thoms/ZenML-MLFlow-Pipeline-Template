# Production-Grade ML Pipeline Template

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ZenML](https://img.shields.io/badge/Orchestration-ZenML-blueviolet)](https://zenml.io/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org/)

A house price prediction project built not to practice linear regression, but to practice **production-grade software engineering and MLOps**.

## 🚀 About This Project

The "Ames Housing" dataset is a classic for a reason, but most implementations stop at the model. This project intentionally takes that simple problem and uses it as a canvas to build a **robust, scalable, and production-ready ML system.**

The primary goal here was **not** model-tuning, but to build a "plug-and-play" system where any component—from data ingestion to feature engineering—can be easily swapped out and tested.

This repository serves as a template for building clean, maintainable, and scalable machine learning systems.

---

## ✨ Key Features

* **Plug-and-Play Architecture:** Heavily utilizes **Strategy** and **Factory** design patterns. You can add a new data cleaning method or a new model type without refactoring the entire pipeline.
* **Abstraction & Interfaces:** All core components (e.g., `model_building`, `data_splitter`) are built against abstract interfaces, making the codebase clean and easy to extend.
* **End-to-End MLOps Pipeline:** Uses **ZenML** to define and orchestrate a reproducible training and deployment pipeline.
* **Full Experiment Tracking:** Integrated with **MLflow** to log all runs, parameters, metrics, and model artifacts automatically.
* **Scalable & Clean Structure:** The project is organized for collaboration, with a clear separation of concerns between `pipelines`, `src` (core logic), and `steps`.

---

## 🛠️ Tech Stack

* **Orchestration:** [ZenML](https://zenml.io/)
* **Experiment Tracking:** [MLflow](https://mlflow.org/)
* **Core Libraries:** Python 3.10+, scikit-learn, Pandas
* **Data:** Ames Housing Dataset

---
Here are some GitHub repo name ideas and a complete README file, all designed to highlight the engineering and MLOps focus of your project.

-----

### Repository Name Ideas

The key is to signal that this is more than just a simple model.

  * **Pluggable-ML-Pipeline:** Highlights the "plug-and-play" aspect you mentioned.
  * **ML-System-Design-Patterns:** Puts the software engineering concept front and center.
  * **Production-Grade-ML-Template:** Shows this is a reusable template for building real-world projects.
  * **Scalable-ML-Architecture-Demo:** Uses the "Ames Housing" problem as a demo for a larger concept.
  * **ZenML-MLFlow-Pipeline-Template:** Clearly states the core MLOps tools used.

-----

### README.md File

Here is a comprehensive README.md file you can copy and paste. It's written in Markdown.

```markdown
# Production-Grade ML Pipeline Template

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ZenML](https://img.shields.io/badge/Orchestration-ZenML-blueviolet)](https://zenml.io/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org/)

A house price prediction project built not to practice linear regression, but to practice **production-grade software engineering and MLOps**.

## 🚀 About This Project

The "Ames Housing" dataset is a classic for a reason, but most implementations stop at the model. This project intentionally takes that simple problem and uses it as a canvas to build a **robust, scalable, and production-ready ML system.**

The primary goal here was **not** model-tuning, but to build a "plug-and-play" system where any component—from data ingestion to feature engineering—can be easily swapped out and tested.

This repository serves as a template for building clean, maintainable, and scalable machine learning systems.

---

## ✨ Key Features

* **Plug-and-Play Architecture:** Heavily utilizes **Strategy** and **Factory** design patterns. You can add a new data cleaning method or a new model type without refactoring the entire pipeline.
* **Abstraction & Interfaces:** All core components (e.g., `model_building`, `data_splitter`) are built against abstract interfaces, making the codebase clean and easy to extend.
* **End-to-End MLOps Pipeline:** Uses **ZenML** to define and orchestrate a reproducible training and deployment pipeline.
* **Full Experiment Tracking:** Integrated with **MLflow** to log all runs, parameters, metrics, and model artifacts automatically.
* **Scalable & Clean Structure:** The project is organized for collaboration, with a clear separation of concerns between `pipelines`, `src` (core logic), and `steps`.

---

## 🛠️ Tech Stack

* **Orchestration:** [ZenML](https://zenml.io/)
* **Experiment Tracking:** [MLflow](https://mlflow.org/)
* **Core Libraries:** Python 3.10+, scikit-learn, Pandas
* **Data:** Ames Housing Dataset

---

## 📁 Project Structure

The repository is structured to be scalable and maintainable.

```

.
├── data/
│   └── extracted\_data/
│       └── AmesHousing.csv
├── .mlruns/                \# MLflow tracking directory
├── pipelines/
│   ├── deployment\_pipeline.py
│   └── training\_pipeline.py  \# ZenML training pipeline definition
├── src/
│   ├── data\_splitter.py
│   ├── feature\_engineering.py
│   ├── handle\_missing\_values.py
│   ├── ingest\_data.py
│   ├── model\_building.py
│   ├── model\_evaluator.py
│   ├── outlier\_detection.py
│   └── steps/                \# ZenML steps
│       ├── data\_ingestion\_step.py
│       ├── data\_splitter\_step.py
│       ├── feature\_engineering\_step.py
│       ├── handle\_missing\_values\_step.py
│       ├── model\_building\_step.py
│       ├── model\_evaluator\_step.py
│       └── outlier\_detection\_step.py
├── tests/
├── config.yaml             \# Configuration file
├── requirements.txt        \# Project dependencies
├── run\_deployment.py       \# Script to run deployment pipeline
├── run\_pipeline.py         \# Script to run training pipeline
└── sample\_predict.py       \# Example script for prediction

````

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher
* [Git](https://git-scm.com/)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Running the Pipeline

1.  **Initialize ZenML:**
    (You may need to initialize ZenML and set up your stack, e.g., `zenml init`)

2.  **Run the training pipeline:**
    This script will execute the ZenML pipeline defined in `pipelines/training_pipeline.py`.
    ```sh
    python run_pipeline.py
    ```

3.  **View experiments in MLflow:**
    Launch the MLflow UI to see your tracked experiments, parameters, and metrics.
    ```sh
    mlflow ui
    ```

4.  **Run the deployment pipeline:**
    (This step will vary based on your `deployment_pipeline.py` logic)
    ```sh
    python run_deployment.py
    ```

---
## 🤝 How to Contribute

This project is built on a "plug-and-play" architecture using **Software Design Patterns**, making it easy to extend and test new components. The core philosophy is to use Abstract Base Classes (Interfaces) for each part of the pipeline, allowing new "strategies" to be added without changing the existing pipeline logic.

### Example: Adding a New Feature Engineering Strategy

Let's use `src/feature_engineering.py` as an example. This file uses the **Strategy Design Pattern**.

1.  **The Interface (Abstract Base Class):**
    The file defines an abstract class, `FeatureEngineeringStrategy`, which has one abstract method: `apply_transformation`.

    ```python
    from abc import ABC, abstractmethod

    class FeatureEngineeringStrategy(ABC):
        @abstractmethod
        def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
            pass
    ```

2.  **The Concrete Strategies:**
    Classes like `LogTransformation`, `StandardScaling`, and `OneHotEncoding` are *concrete implementations* of this interface. They all inherit from `FeatureEngineeringStrategy` and provide their own logic for the `apply_transformation` method.

3.  **How to Add Your Own Strategy (e.g., `NewCustomScaling`):**

    * **Step 1:** Open the relevant file (e.g., `src/feature_engineering.py`).
    * **Step 2:** Create your new class, inheriting from the `FeatureEngineeringStrategy` base class.
    * **Step 3:** Implement the `apply_transformation` method with your custom logic.

    ```python
    # Add this class to src/feature_engineering.py

    from sklearn.preprocessing import RobustScaler # Or any other tool you need

    # ... (other imports)

    # Concrete Strategy for a NEW Custom Scaling
    # ----------------------------------------
    class NewCustomScaling(FeatureEngineeringStrategy):
        def __init__(self, features):
            self.features = features
            self.scaler = RobustScaler() # Using RobustScaler as an example

        def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
            logging.info(f"Applying NEW custom scaling to features: {self.features}")
            df_transformed = df.copy()
            df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
            logging.info("NEW custom scaling completed.")
            return df_transformed
    ```

    * **Step 4:** You can now import and use `NewCustomScaling` in your ZenML pipeline steps (e.g., `src/steps/feature_engineering_step.py`) just by swapping out the class that gets instantiated.

This same principle applies to `model_building`, `data_splitter`, and other core components. This approach keeps the codebase clean, decoupled, and easy for anyone to test new ideas.

---

## LICENSE

This project is licensed under the MIT License. See the `LICENSE` file for details.
````
