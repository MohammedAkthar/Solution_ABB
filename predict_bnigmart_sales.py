import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import time
import warnings
import matplotlib.pyplot as plt

# from data_preparation import DataPreparation

warnings.filterwarnings('ignore')


class DataPreparation:
    """
    Handles data preparation
    Prevents data leakage by fitting on train, transforming on both
    """

    def __init__(self):
        self.imputers = {}
        self.encoders = {}
        self.scalers = {}
        self.bin_edges = {}

    def transform_data(self, df):
        """
        Transforming Item Fat content Data
        """
        df_transform = df.copy()

        df_transform['Item_Fat_Content'] = df_transform['Item_Fat_Content'].replace('LF', 'Low Fat')
        df_transform['Item_Fat_Content'] = df_transform['Item_Fat_Content'].replace('low fat', 'Low Fat')
        df_transform['Item_Fat_Content'] = df_transform['Item_Fat_Content'].replace('reg', 'Regular')

        print("\n Transformed Item Fat Content: \n", df_transform['Item_Fat_Content'].unique())

        # Marking zero visibility as missing (will impute later)
        df_transform.loc[df_transform['Item_Visibility'] == 0, 'Item_Visibility'] = np.nan
        return df_transform

    def feature_engineering(self, df):
        """
        Creating additional features from given dataset
        """
        df_features = df.copy()

        df_features['Item_Category'] = df_features['Item_Identifier'].str[:2]

        # Calculate outlet age (deterministic)
        current_year = 2025
        df_features['Outlet_Age'] = current_year - df_features['Outlet_Establishment_Year']
        return df_features

    def fit_imputers(self, train_df):
        """
        FIT imputers on training data only
        """

        # Item_Weight imputer by Item_Type
        self.imputers['item_weight'] = train_df.groupby('Item_Type')['Item_Weight'].mean().to_dict()

        # Item_Visibility imputer by Item_Identifier
        self.imputers['item_visibility_by_id'] = train_df.groupby('Item_Identifier')['Item_Visibility'].mean().to_dict()
        self.imputers['item_visibility_overall'] = train_df['Item_Visibility'].mean()

        # Outlet_Size imputer by Outlet_Type
        outlet_size_mode = {}
        for outlet_type in train_df['Outlet_Type'].unique():
            mask = train_df['Outlet_Type'] == outlet_type
            mode_series = train_df.loc[mask, 'Outlet_Size'].mode()
            outlet_size_mode[outlet_type] = mode_series[0] if len(mode_series) > 0 else 'Medium'
        self.imputers['outlet_size'] = outlet_size_mode
        return self

    def transform_imputation(self, df):
        """
        TRANSFORM imputers
        """
        df_imputed = df.copy()

        missing_values = df_imputed.isnull().sum().sum()

        # Impute Item_Weight
        df_imputed['Item_Weight'] = df_imputed.apply(
            lambda row: self.imputers['item_weight'].get(row['Item_Type'], df_imputed['Item_Weight'].mean()) if pd.isna(row['Item_Weight']) else row['Item_Weight'],
            axis=1
            )

        # Impute Item_Visibility
        df_imputed['Item_Visibility'] = df_imputed.apply(
            lambda row: self.imputers['item_visibility_by_id'].get(row['Item_Identifier'],self.imputers['item_visibility_overall']) if pd.isna(row['Item_Visibility']) else row['Item_Visibility'],
            axis=1
            )

        # Impute Outlet_Size
        df_imputed['Outlet_Size'] = df_imputed.apply(
            lambda row: self.imputers['outlet_size'].get(row['Outlet_Type'], 'Medium') if pd.isna(row['Outlet_Size']) else row['Outlet_Size'],
            axis=1
            )

        imputed_values = df_imputed.isnull().sum().sum()

        print(f"Missing values before: {missing_values}")
        print(f"Missing values after: {imputed_values}")

        return df_imputed

    def fit_binning(self, train_df):
        """
        FIT binning on training data
        Determine bin edges from training set
        """

        # Price categories - determine bins from train
        _, self.bin_edges['price'] = pd.qcut(
            train_df['Item_MRP'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Premium'],
            retbins=True,
            duplicates='drop'
        )

        # Visibility categories - determine bins from train
        _, self.bin_edges['visibility'] = pd.qcut(
            train_df['Item_Visibility'],
            q=3,
            labels=['Low_Vis', 'Medium_Vis', 'High_Vis'],
            retbins=True,
            duplicates='drop'
        )

        # Store age categories (fixed bins, but good practice to define)
        self.bin_edges['store_age'] = [0, 5, 15, 30]
        return self

    def transform_binning(self, df):
        """
        TRANSFORM using fitted bins
        """
        df_binned = df.copy()

        # Apply price categories using train bins
        df_binned['Price_Category'] = pd.cut(
            df_binned['Item_MRP'],
            bins=self.bin_edges['price'],
            labels=['Low', 'Medium', 'High', 'Premium'],
            include_lowest=True
        )

        # Apply visibility categories using train bins
        df_binned['Visibility_Category'] = pd.cut(
            df_binned['Item_Visibility'],
            bins=self.bin_edges['visibility'],
            labels=['Low_Vis', 'Medium_Vis', 'High_Vis'],
            include_lowest=True
        )

        # Apply store age categories
        df_binned['Store_Age_Category'] = pd.cut(
            df_binned['Outlet_Age'],
            bins=self.bin_edges['store_age'],
            labels=['New', 'Established', 'Mature']
        )


        return df_binned

    def fit_encoders(self, train_df):
        """
        FIT encoders on training data
        """
        categorical_columns = [
            'Item_Identifier', 'Item_Fat_Content', 'Item_Type',
            'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
            'Outlet_Type',
            'Item_Category',
            'Price_Category',
            'Visibility_Category', 'Store_Age_Category'
        ]

        for col in categorical_columns:
            if col in train_df.columns:
                le = LabelEncoder()
                # Fit on train data only
                le.fit(train_df[col].astype(str))
                self.encoders[col] = le

        return self

    def transform_encoding(self, df):
        """
        TRANSFORM using fitted encoders
        """

        df_encoded = df.copy()

        for col, encoder in self.encoders.items():
            if col in df_encoded.columns:
                # Handle unseen categories in test set
                df_encoded[col] = df_encoded[col].astype(str)

                # For values not seen in training, assign a default
                known_classes = set(encoder.classes_)
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in known_classes else encoder.classes_[0]
                )

                df_encoded[col] = encoder.transform(df_encoded[col])

        return df_encoded

    def fit_scalers(self, train_df):
        """
        FIT scalers on training data
        """

        scale_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']

        scaler = StandardScaler()
        scaler.fit(train_df[scale_features])
        self.scalers['standard'] = scaler
        self.scalers['features'] = scale_features

        print(f"✓ StandardScaler fitted on {len(scale_features)} features")

        return self

    def transform_scaling(self, df):
        """
        TRANSFORM using fitted scalers
        """

        df_scaled = df.copy()

        features = self.scalers['features']
        df_scaled[features] = self.scalers['standard'].transform(df_scaled[features])

        return df_scaled


class ModelTrainer:
    """
    Model Training and Evaluation
    """

    def __init__(self, random_state=42):
        """
        Initialize ModelTrainer
        """
        self.random_state = random_state
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.feature_importance = {}

    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None, tune_hyperparams=False):
        """
        Train Random Forest Regressor
        """

        print("TRAINING RANDOM FOREST")
        start_time = time.time()
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': self.random_state,
            'n_jobs': -1
                        }

        # Hyperparameter tuning
        if tune_hyperparams:
            print("\nPerforming hyperparameter tuning...")
            model = self._tune_random_forest(X_train, y_train)
        else:
            print(f"\nUsing parameters: {default_params}")
            model = RandomForestRegressor(**default_params)
            model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Store model
        self.models['random_forest'] = model

        # Evaluate
        if X_test is not None and y_test is not None:
            self._evaluate_model('random_forest', model, X_train, y_train,
                                 X_test, y_test, training_time)
        return model

    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None,
                       tune_hyperparams=False):
        """
        Train XGBoost Regressor
        """
        start_time = time.time()
        default_params = {
                'n_estimators': 100,
                'max_depth': 7,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1
            }

        if tune_hyperparams:
            print("\n Performing hyperparameter tuning...")
            model = self._tune_xgboost(X_train, y_train)
        else:
            print(f"\nUsing parameters: {default_params}")
            model = xgb.XGBRegressor(**default_params)
            model.fit(X_train, y_train)

        model_name = 'xgboost'
        training_time = time.time() - start_time

        # Store model
        self.models[model_name] = model
        # Evaluate
        if X_test is not None and y_test is not None:
            self._evaluate_model(model_name, model, X_train, y_train,
                                 X_test, y_test, training_time)
        return model


    def _tune_random_forest(self, X_train, y_train, cv=3):
        """Tune Random Forest hyperparameters using RandomizedSearchCV"""
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }

        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)

        random_search = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state, n_jobs=-1, verbose=1
        )

        random_search.fit(X_train, y_train)

        print(f"\n Best parameters: {random_search.best_params_}")
        print(f" Best CV RMSE: {np.sqrt(-random_search.best_score_):.2f}")

        return random_search.best_estimator_

    def _tune_xgboost(self, X_train, y_train, cv=3):
        """Tune XGBoost hyperparameters using RandomizedSearchCV"""
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        xgb_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)

        random_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=20, cv=cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state, n_jobs=-1, verbose=1
        )

        random_search.fit(X_train, y_train)

        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.2f}")

        return random_search.best_estimator_


    def generate_predictions(self, model, X_test,prep):
        """
        Generate Predictions
        """
        # Predictions
        y_test_pred = model.predict(X_test)

        # Store predictions
        X_test['Item_Outlet_Sales'] = y_test_pred

        columns_to_export = ["Item_Identifier","Outlet_Identifier","Item_Outlet_Sales"]
        final_df = X_test.loc[:, columns_to_export]

        encode_cols = ["Item_Identifier", "Outlet_Identifier"]

        for col in encode_cols:
            label_encoders = prep.encoders[col]
            final_df[col] = label_encoders.inverse_transform(X_test[col])

        final_df.loc[final_df['Item_Outlet_Sales'] < 0, 'Item_Outlet_Sales'] = 0

        final_df.to_csv('big-mart-submission.csv',index=False)

        return final_df


    def _evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test, training_time):
        """
        Evaluate model performance on train and test sets
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Store predictions
        self.predictions[model_name] = {
            'train': y_train_pred,
            'test': y_test_pred
        }

        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
            'training_time': training_time
        }

        # Store metrics
        self.metrics[model_name] = metrics

        # Display metrics
        print(f"\n {model_name.upper().replace('_', ' ')} PERFORMANCE METRICS:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Training':<15} {'Test':<15}")
        print("-" * 70)
        print(f"{'RMSE':<20} {metrics['train_rmse']:>12,.2f}   {metrics['test_rmse']:>12,.2f}")
        print(f"{'MAE':<20} {metrics['train_mae']:>12,.2f}   {metrics['test_mae']:>12,.2f}")
        print(f"{'R² Score':<20} {metrics['train_r2']:>12.4f}   {metrics['test_r2']:>12.4f}")
        print(f"{'MAPE (%)':<20} {metrics['train_mape']:>12.2f}   {metrics['test_mape']:>12.2f}")
        print("-" * 70)


        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = model.feature_importances_

    def plot_feature_importance(self, model_name, feature_names, top_n=20):
        """
        Plot feature importance for a model

        """
        if model_name not in self.feature_importance:
            print(f" Feature importance not available for '{model_name}'")
            return

        importance = self.feature_importance[model_name]

        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df)), importance_df['Importance'], color='steelblue')
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{model_name.replace("_", " ").title()} - Top {top_n} Features',
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✓ Feature importance plot saved as '{model_name}_feature_importance.png'")

        return importance_df

def prepare_data(train_path, test_path):
    """
    Run Pipeline
    """

    # Load pre-split data
    print("\n Loading train and test data...")

    train_df = pd.read_csv(train_path)
    train_df['split']='train'

    test_df = pd.read_csv(test_path)
    test_df['split'] = 'test'

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Initialize preparation pipeline
    prep = DataPreparation()

    # Combine for cleaning and deterministic feature engineering
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    train_size = len(train_df)

    # Transform Data / Clean Data
    combined_transform = prep.transform_data(combined_df)

    # Feature Engineering
    combined_features = prep.feature_engineering(combined_transform)

    # Split back
    train_df = combined_features[combined_features['split'] == 'train'].copy()
    test_df = combined_features[combined_features['split'] == 'test'].copy()

    train_df = train_df.drop(columns=['split'])
    test_df = test_df.drop(columns=['Item_Outlet_Sales','split'])

    print(f"\n Data cleaned and features engineered on combined dataset")
    print(f"Split back - Train: {train_df.shape}, Test: {test_df.shape}")


    # Imputation
    print("\n Fitting Imputes On Training Data")
    prep.fit_imputers(train_df)

    print("\nTransforming Imputes On Training Data")
    train_df = prep.transform_imputation(train_df)

    print("\nTransforming Imputes On Testing Data")
    test_df = prep.transform_imputation(test_df)

    # Binning
    print("\nFitting Binning On Training Data")
    prep.fit_binning(train_df)

    print("\n Transforming Binning On Training Data")
    train_df = prep.transform_binning(train_df)

    print("\n Transforming Binning On Testing Data")
    test_df = prep.transform_binning(test_df)

    # Encoding
    print("\n Fitting Encoder On Training Data")
    prep.fit_encoders(train_df)

    print("\n Transforming Encoder On Training Data")
    train_df = prep.transform_encoding(train_df)
    print(train_df.head())

    print("\n Transforming Encoder On Testing Data")
    test_df = prep.transform_encoding(test_df)


    prep.fit_scalers(train_df)
    train_df = prep.transform_scaling(train_df)
    print(train_df.head())

    test_df = prep.transform_scaling(test_df)


    print(f"\nFinal shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")

    return train_df, test_df, prep



if __name__ == "__main__":
    train_path = 'train.csv'
    test_path = 'test.csv'
    train_df, test_df, prep = prepare_data(train_path, test_path)

    train_df.to_csv('train_processed.csv',index=False)
    test_df.to_csv('test_processed.csv', index=False)

    train_size = int(0.8 * len(train_df))

    train_df_1 = train_df.iloc[:train_size].copy()
    test_df_1 = train_df.iloc[train_size:].copy()

    y_train = train_df_1["Item_Outlet_Sales"]
    X_train = train_df_1.drop(columns=["Item_Outlet_Sales"])

    y_test = test_df_1["Item_Outlet_Sales"]
    X_test = test_df_1.drop(columns=["Item_Outlet_Sales"])

    # Initialize trainer
    trainer = ModelTrainer(random_state=42)

    # Train models
    # rf_model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model = trainer.train_xgboost(X_train, y_train, X_test, y_test)


    """
    # Adding  hyperparameter tuning
    rf_tuned = trainer.train_random_forest(
        X_train, y_train, X_test, y_test, 
        tune_hyperparams=True
    )
    trainer.generate_predictions(rf_tuned, test_df, prep)
    """


    # Adding  hyperparameter tuning
    xgb_tuned = trainer.train_xgboost(
            X_train, y_train, X_test, y_test,
            tune_hyperparams=True
        )
    trainer.generate_predictions(xgb_tuned, test_df, prep)

    # Feature importance
    importance = trainer.plot_feature_importance('xgboost',
                                                 X_train.columns, top_n=20)
