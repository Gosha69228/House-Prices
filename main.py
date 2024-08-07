import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb


def preprocess_data(input_file):
    df = pd.read_csv(input_file)

    # заполнение пропусков
    # обработка очень малого кол-ва пропусков (1-4)
    columns_to_fill_mode = ['SaleType', 'Functional', 'KitchenQual', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath',
                            'Exterior2nd', 'Exterior1st', 'Utilities', 'MSZoning', 'GarageCars', 'BsmtFinSF2',
                            'BsmtFinSF1']
    fill_values = {col: df[col].mode()[0] for col in columns_to_fill_mode}
    df.fillna(fill_values, inplace=True)

    columns_to_fill_median = ['GarageArea', 'TotalBsmtSF', 'BsmtUnfSF']
    fill_values2 = {col: df[col].median() for col in columns_to_fill_median}
    df.fillna(fill_values2, inplace=True)

    # обработка большого кол-ва пропусков
    df.Alley = df.Alley.fillna('NA')
    df.MasVnrType = df.MasVnrType.fillna('None')
    df.PoolQC = df.PoolQC.fillna('NA')
    df.Fence = df.Fence.fillna('NA')
    df.MiscFeature = df.MiscFeature.fillna('NA')
    df.FireplaceQu = df.FireplaceQu.fillna('NA')
    df.MasVnrArea = df.MasVnrArea.fillna(0)

    df.BsmtQual = df.BsmtQual.fillna('NA')
    for col in ['BsmtExposure', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']:
        df.loc[df['BsmtQual'] == 'NA', col] = 'NA'

    columns_to_fill_mode2 = ['BsmtFinType2', 'BsmtCond', 'BsmtExposure']
    fill_values = {col: df[col].mode()[0] for col in columns_to_fill_mode2}
    df.fillna(fill_values, inplace=True)

    df.GarageType = df.GarageType.fillna('NA')
    for col in ['GarageFinish', 'GarageQual', 'GarageCond']:
        df.loc[df['GarageType'] == 'NA', col] = 'NA'
    for col in ['GarageYrBlt']:
        df.loc[df['GarageType'] == 'NA', col] = 0

    columns_to_fill_mode2 = ['GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt']
    fill_values = {col: df[col].mode()[0] for col in columns_to_fill_mode2}
    df.fillna(fill_values, inplace=True)

    # Разделение данных на строки с известными и неизвестными значениями LotFrontage
    known_frontage = df[df['LotFrontage'].notna()]
    unknown_frontage = df[df['LotFrontage'].isna()]
    # Подготовка данных для модели
    X_train = known_frontage[['LotArea']]
    y_train = known_frontage['LotFrontage']
    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Предсказание значений LotFrontage для строк с пропусками
    X_test = unknown_frontage[['LotArea']]
    y_pred = model.predict(X_test)
    # Заполнение пропусков в LotFrontage предсказанными значениями
    df.loc[df['LotFrontage'].isna(), 'LotFrontage'] = y_pred

    # преобразование порядковых признаков
    ordinal_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                       'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    encoder = LabelEncoder()
    for col in ordinal_columns:
        df[col + '_encoded'] = encoder.fit_transform(df[col])
    # откидываем преобразованные колонки
    df = df.drop(columns=ordinal_columns)

    # преобразование категориальных признаков
    categories_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                          'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                          'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                          'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence',
                          'MiscFeature', 'SaleType', 'SaleCondition']
    # Применяем One-Hot Encoding ко всем категориальным признакам
    df = pd.get_dummies(df, columns=categories_columns)

    # нормализация и стандартизация данных
    data_to_standard = df[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                           'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                           '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']]
    standard_scaler = StandardScaler()
    standardized_data = standard_scaler.fit_transform(data_to_standard)

    processed_df = pd.DataFrame(
        np.hstack([standardized_data]),
        columns=['LotFrontage_standardized', 'LotArea_standardized', 'YearBuilt_standardized',
                 'YearRemodAdd_standardized', 'MasVnrArea_standardized', 'BsmtFinSF1_standardized',
                 'BsmtFinSF2_standardized', 'BsmtUnfSF_standardized', 'TotalBsmtSF_standardized',
                 '1stFlrSF_standardized', '2ndFlrSF_standardized', 'LowQualFinSF_standardized', 'GrLivArea_standardized',
                 'GarageYrBlt_standardized', 'GarageArea_standardized', 'WoodDeckSF_standardized',
                 'OpenPorchSF_standardized', 'EnclosedPorch_standardized', '3SsnPorch_standardized', 'ScreenPorch_standardized',
                 'PoolArea_standardized', 'MiscVal_standardized', 'YrSold_standardized']
    )
    columns_to_drop = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                       'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                       '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']
    df.drop(columns=columns_to_drop, inplace=True)

    combined_df = pd.concat([df.reset_index(drop=True), processed_df.reset_index(drop=True)], axis=1)
    return combined_df

best_params = {
    'colsample_bytree': 0.5,
    'gamma': 0,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 3
}
# Создаем объект модели XGBoost
model = xgb.XGBRegressor(**best_params, random_state=42)

input_file = "train.csv"
preprocess_file = preprocess_data(input_file)

input_file2 = "output.csv"
preprocess_file2 = preprocess_data(input_file2)
known_Price = preprocess_file2[preprocess_file2['SalePrice'].notnull()]
unknown_Price = preprocess_file2[preprocess_file2['SalePrice'].isnull()].drop(columns=['SalePrice'])

X_train_Price = known_Price.drop(columns=['SalePrice'])
y_train_Price = known_Price['SalePrice']

rf_Price = xgb.XGBRegressor(**best_params, random_state=42)
rf_Price.fit(X_train_Price, y_train_Price)

predicted_Price = rf_Price.predict(unknown_Price)

preprocess_file2.loc[preprocess_file2['SalePrice'].isnull(), 'SalePrice'] = predicted_Price
preprocess_file2 = preprocess_file2[['Id', 'SalePrice']]
preprocess_file2.SalePrice = preprocess_file2.SalePrice.astype(float)
preprocess_file2.iloc[1460:].to_csv('kag.csv', index=False)
