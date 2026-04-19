import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class OrbitalPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median") 
        self.expected_columns = None
        
    def _data_clean(self, df):
        # --- FEATURE ENGINEERING ---
        df_sorted = df.sort_values(by=['event_id', 'time_to_tca'], ascending=[True, False])
        df_early = df_sorted[df_sorted['time_to_tca'] >= 2.0]
        clean_df = df_early.groupby('event_id').last().reset_index()
        
        has_target = 'risk' in clean_df.columns
        if has_target:
            y = (clean_df['risk'] >= -4.0).astype(int)
        else:
            y = None

        # --- COLUMN DROPS ---
        clean_df = clean_df.drop(['relative_position_r', 'relative_position_t', 'relative_position_n', 
                                'relative_velocity_r', 'relative_velocity_t', 'relative_velocity_n',
                                'time_to_tca'], axis=1, errors='ignore')
        
        angles = ['geocentric_latitude', 'azimuth', 'elevation', 't_j2k_inc', 'c_j2k_inc']
        for col in angles:
            clean_df[f'{col}_sin'] = np.sin(np.radians(clean_df[col]))
            clean_df[f'{col}_cos'] = np.cos(np.radians(clean_df[col]))
            clean_df = clean_df.drop(columns=[col])

        leakage_features = ['event_id', 'mission_id', 'risk', 'max_risk_estimate', 'max_risk_scaling']
        clean_df = clean_df.drop(columns=leakage_features, errors='ignore')      

        meta_cols = [col for col in clean_df.columns if 'lastob' in col or 'obs_' in col or 'residuals' in col or 'od_span' in col or 'rms' in col]
        clean_df = clean_df.drop(columns=meta_cols)

        cross_terms = [col for col in clean_df.columns if (col.startswith('t_c') or col.startswith('c_c')) and 'type' not in col and 'span' not in col and 'covariance' not in col]
        clean_df = clean_df.drop(columns=cross_terms, errors='ignore')

        # --- ONE-HOT ENCODING ---
        X = pd.get_dummies(clean_df, columns=['c_object_type'])
        
        if self.expected_columns is not None:
            for col in self.expected_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.expected_columns]

        return X, y

    def fit_transform_train(self, df_train):
        X, y = self._data_clean(df_train)
        self.expected_columns = X.columns.tolist()
        X_imputed = self.imputer.fit_transform(X)
        X_scaled_array = self.scaler.fit_transform(X_imputed)
        X_train_scaled = pd.DataFrame(X_scaled_array, columns=self.expected_columns)

        return X_train_scaled , y

    def transform_new_data(self, df_new):
        X, y = self._data_clean(df_new)
        X_imputed = self.imputer.transform(X)
        X_scaled_array = self.scaler.transform(X_imputed)
        X_test_scaled = pd.DataFrame(X_scaled_array, columns=self.expected_columns)

        return X_test_scaled, y

