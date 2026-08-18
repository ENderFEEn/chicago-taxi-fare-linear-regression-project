# Chicago Taxi Fare Prediction with Linear Regression

This project builds a multiple linear regression model to estimate taxi fares from trip distance and trip duration. I have decided to implement the model directly with NumPy rather than using a ready-made regression estimator from a machine learning library.

The notebook covers the full workflow - loading and inspecting the data, exploratory analysis, train/test splitting, feature standardization, gradient-descent training, evaluation, diagnostic plots, and prediction for new trips.

## Dataset

The notebook uses the open-source Chicago taxi training dataset published for Google's Machine Learning Crash Course:

`https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv`

The data is loaded directly from the URL, so a separate CSV file is not required in the repository.

The notebook keeps the following columns for analysis:

- `TRIP_MILES`
- `TRIP_SECONDS`
- `FARE`
- `COMPANY`
- `PAYMENT_TYPE`
- `TIP_RATE`

`TRIP_MINUTES` is derived from `TRIP_SECONDS` and used as a model feature.

After replacing infinite values and removing missing values from the model columns, 31,694 observations remain.

## Model

The regression model uses two input features:

- trip distance in miles
- trip duration in minutes

The target variable is `FARE`.

The implementation includes:

- a custom 80/20 train/test split
- feature standardization fitted only on the training set
- a linear model with weights and bias initialized to zero
- batch gradient descent using mean squared error
- 300 training epochs
- learning rate of `0.01`

The model is trained without `scikit-learn`. The split, scaler, training loop, prediction logic, and regression metrics are implemented in the notebook.

## Results

The final model produced the following metrics:

| Dataset | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Train | 1.5271 | 14.0129 | 3.7434 | 0.9515 |
| Test | 1.4726 | 11.3958 | 3.3758 | 0.9598 |

Converted back from standardized features, the fitted model is approximately:

```text
FARE = 1.7629 × TRIP_MILES
     + 0.2715 × TRIP_MINUTES
     + 3.2710
```

The test-set R² of about 0.96 indicates that trip distance and duration explain most of the fare variation in this dataset. The remaining error also shows that these two variables do not capture every factor that affects a taxi fare.

## Analysis and Diagnostics

The notebook includes several plots used to inspect the data and model behavior:

- fare versus trip distance
- fare versus trip duration
- correlation matrix
- RMSE over training epochs
- actual versus predicted fares
- residuals versus predicted fares
- selected actual and predicted fare values
- 3D regression plane for distance, duration, and fare

The exploratory analysis shows a strong relationship between fare and trip distance, with a correlation of approximately `0.975`. Trip duration also has a strong positive relationship with fare, with a correlation of approximately `0.830`.

## Predicting New Trips

The notebook contains a helper function for estimating fares for new trips. New observations are supplied with:

```text
TRIP_MILES
TRIP_SECONDS
```

The function converts seconds to minutes, applies the scaler learned from the training data, and passes the transformed values to the trained model.

Example input:

```python
new_trips = pd.DataFrame({
    "TRIP_MILES": [4.5, 1.2, 8.0],
    "TRIP_SECONDS": [700, 1200, 1100]
})
```

The corresponding predictions from the trained model are approximately:

| Trip Miles | Trip Seconds | Predicted Fare |
|---:|---:|---:|
| 4.5 | 700 | 14.37 |
| 1.2 | 1200 | 10.82 |
| 8.0 | 1100 | 22.35 |

## Running the Notebook

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open `linear_regression_taxi_fare.ipynb` and run the cells in order.

An internet connection is required when the dataset is downloaded for the first time.

## Project Structure

```text
chicago-taxi-fare-linear-regression/
|- linear_regression_taxi_fare.ipynb
|- README.md
|- requirements.txt
```

## Limitations

This project is intended to demonstrate the mechanics of linear regression and the supporting data workflow. It does not serve as a production fare-prediction system.

The current model uses only distance and duration. It does not use potentially relevant information such as taxi company, payment type, traffic conditions, time of day, pickup/drop-off location, or pricing rules. The dataset is also randomly split rather than evaluated using a time-based validation strategy.

These choices keep the implementation focused on understanding linear regression from first principles.
