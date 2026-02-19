


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

# ====================== ГЕНЕРАЦИЯ ДАННЫХ ======================

class DataGenerator:
    def __init__(self):
        self.regions = ['DE', 'FR', 'PL', 'RO', 'IT', 'ES', 'HU', 'BG']
        self.base_yield = {'DE': 70, 'FR': 75, 'PL': 55, 'RO': 45, 
                          'IT': 60, 'ES': 50, 'HU': 58, 'BG': 48}
        np.random.seed(42)
    
    def generate_weather_data(self):
        """Генерация метеоданных и вегетационных индексов"""
        data = []
        for region in self.regions:
            for year in range(2010, 2023):
                for month in range(1, 13):
                    # Метеоданные с сезонностью
                    temp = 10 + 15 * np.sin(2*np.pi*(month-4)/12) + np.random.normal(0, 2)
                    precip = 50 + 30 * np.cos(2*np.pi*(month-2)/12) + np.random.normal(0, 10)
                    
                    # Вегетационные индексы
                    ndvi = 0.3 + 0.5 * np.sin(2*np.pi*(month-4)/12) + np.random.normal(0, 0.05)
                    ndvi = np.clip(ndvi, 0.2, 0.9)
                    evi = ndvi * 1.2 + np.random.normal(0, 0.03)
                    
                    data.append({
                        'region': region, 'year': year, 'month': month,
                        'temp': temp, 'temp_max': temp + 5, 'temp_min': temp - 5,
                        'precip': precip, 'humidity': 60 + 15*np.sin(2*np.pi*month/12),
                        'pressure': 1013 + np.random.normal(0, 5),
                        'wind': 4 + 2*np.sin(2*np.pi*month/12),
                        'solar': 200 + 100*np.sin(2*np.pi*(month-6)/12),
                        'ndvi': ndvi, 'evi': evi
                    })
        return pd.DataFrame(data)
    
    def generate_yield_data(self):
        """Генерация данных урожайности"""
        data = []
        for region in self.regions:
            base = self.base_yield[region]
            for year in range(2010, 2023):
                # Тренд + случайность
                yield_val = base + 0.3*(year-2010) + np.random.normal(0, 3)
                data.append({'region': region, 'year': year, 'yield': yield_val})
        return pd.DataFrame(data)
    
    def create_dataset(self):
        """Объединение данных"""
        weather = self.generate_weather_data()
        yields = self.generate_yield_data()
        
        # Добавляем целевую переменную
        df = weather.merge(yields, on=['region', 'year'])
        
        # Добавляем временные признаки
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)
        
        return df


# ====================== ПРЕДОБРАБОТКА ======================

class Preprocessor:
    def __init__(self, seq_length=12):
        self.seq_length = seq_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_cols = ['temp', 'temp_max', 'temp_min', 'precip', 'humidity',
                            'pressure', 'wind', 'solar', 'ndvi', 'evi', 
                            'month_sin', 'month_cos']
    
    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i+self.seq_length])
            y.append(target[i+self.seq_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df):
        """Подготовка данных для обучения"""
        X_list, y_list, regions_list = [], [], []
        
        for region in df['region'].unique():
            region_df = df[df['region']==region].sort_values(['year', 'month'])
            
            # Масштабирование
            X_scaled = self.scaler_X.fit_transform(region_df[self.feature_cols])
            y_scaled = self.scaler_y.fit_transform(region_df[['yield']]).flatten()
            
            # Создание последовательностей
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
            
            if len(X_seq) > 0:
                X_list.append(X_seq)
                y_list.append(y_seq)
                regions_list.extend([region]*len(X_seq))
        
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        regions = np.array(regions_list)
        
        return X, y, regions
    
    def split_data(self, X, y, regions):
        """Разделение на train/val/test"""
        X_train, X_temp, y_train, y_temp, r_train, r_temp = train_test_split(
            X, y, regions, test_size=0.3, random_state=42, stratify=regions
        )
        X_val, X_test, y_val, y_test, r_val, r_test = train_test_split(
            X_temp, y_temp, r_temp, test_size=0.5, random_state=42, stratify=r_temp
        )
        return (X_train, y_train, r_train), (X_val, y_val, r_val), (X_test, y_test, r_test)


# ====================== МОДЕЛИ ======================

def build_lstm(input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, 'relu'),
        layers.Dense(1)
    ])
    return model

def build_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(32)(inputs)
    
    # Transformer блок
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, 'relu')(x)
    outputs = layers.Dense(1)(x)
    
    return Model(inputs, outputs)

def build_cnn_lstm(input_shape):
    model = tf.keras.Sequential([
        layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, 'relu'),
        layers.Dense(1)
    ])
    return model


# ====================== ОБУЧЕНИЕ И ОЦЕНКА ======================

class Trainer:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30):
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=0
        )
        return self.history
    
    def evaluate(self, X_test, y_test, scaler_y):
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}, y_true, y_pred


# ====================== ВИЗУАЛИЗАЦИЯ ======================

class Visualizer:
    def plot_results(self, y_true, y_pred, regions, model_name):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot
        axes[0,0].scatter(y_true, y_pred, alpha=0.5)
        axes[0,0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 'r--')
        axes[0,0].set_xlabel('Факт (ц/га)')
        axes[0,0].set_ylabel('Прогноз (ц/га)')
        axes[0,0].set_title(f'{model_name}: Прогноз vs Факт')
        axes[0,0].grid(True)
        
        # 2. Распределение ошибок
        errors = y_true.flatten() - y_pred.flatten()
        axes[0,1].hist(errors, bins=20, edgecolor='black')
        axes[0,1].axvline(0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Ошибка (ц/га)')
        axes[0,1].set_ylabel('Частота')
        axes[0,1].set_title('Распределение ошибок')
        axes[0,1].grid(True)
        
        # 3. MAE по регионам
        regions_unique = np.unique(regions)
        regional_mae = []
        for reg in regions_unique:
            mask = regions == reg
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            regional_mae.append(mae)
        
        bars = axes[1,0].bar(regions_unique, regional_mae)
        axes[1,0].set_xlabel('Регион')
        axes[1,0].set_ylabel('MAE (ц/га)')
        axes[1,0].set_title('MAE по регионам')
        
        # Раскраска по значению
        for bar, val in zip(bars, regional_mae):
            bar.set_color('green' if val < 5 else 'orange' if val < 8 else 'red')
        
        # 4. Пример временного ряда
        reg = regions_unique[0]
        mask = regions == reg
        axes[1,1].plot(y_true[mask][:20], 'o-', label='Факт')
        axes[1,1].plot(y_pred[mask][:20], 's--', label='Прогноз')
        axes[1,1].set_xlabel('Временной шаг')
        axes[1,1].set_ylabel('Урожайность (ц/га)')
        axes[1,1].set_title(f'Пример для региона {reg}')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results_{model_name}.png', dpi=100)
        plt.show()


# ====================== ОСНОВНАЯ ПРОГРАММА ======================

def main():
    print("="*60)
    print("ПРОГНОЗИРОВАНИЕ УРОЖАЙНОСТИ ЗЕРНОВЫХ")
    print("="*60)
    
    # 1. Генерация данных
    print("\n1. Генерация данных...")
    data_gen = DataGenerator()
    df = data_gen.create_dataset()
    print(f"   Создано {len(df)} записей")
    
    # 2. Предобработка
    print("\n2. Предобработка данных...")
    preproc = Preprocessor(seq_length=12)
    X, y, regions = preproc.prepare_data(df)
    (X_train, y_train, r_train), (X_val, y_val, r_val), (X_test, y_test, r_test) = \
        preproc.split_data(X, y, regions)
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 3. Создание и обучение моделей
    print("\n3. Обучение моделей...")
    
    models = {
        'LSTM': build_lstm(X.shape[1:]),
        'Transformer': build_transformer(X.shape[1:]),
        'CNN-LSTM': build_cnn_lstm(X.shape[1:])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   {name} модель:")
        trainer = Trainer(model, name)
        trainer.train(X_train, y_train, X_val, y_val, epochs=30)
        metrics, y_true, y_pred = trainer.evaluate(X_test, y_test, preproc.scaler_y)
        
        print(f"   MAE: {metrics['MAE']:.2f} ц/га, RMSE: {metrics['RMSE']:.2f} ц/га, R2: {metrics['R2']:.3f}")
        
        results[name] = metrics
        
        # Визуализация для лучшей модели
        if name == 'CNN-LSTM':
            vis = Visualizer()
            vis.plot_results(y_true, y_pred, r_test, name)
    
    # 4. Сравнение моделей
    print("\n4. СРАВНЕНИЕ МОДЕЛЕЙ")
    print("-"*40)
    for name, metrics in results.items():
        print(f"{name:12} MAE: {metrics['MAE']:6.2f} | RMSE: {metrics['RMSE']:6.2f} | R2: {metrics['R2']:.3f}")
    
    # 5. Итоговая визуализация
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(results.keys())
    maes = [results[n]['MAE'] for n in names]
    rmses = [results[n]['RMSE'] for n in names]
    r2s = [results[n]['R2'] for n in names]
    
    ax[0].bar(names, maes, color=['blue', 'red', 'green'])
    ax[0].set_title('MAE (ц/га)')
    ax[0].grid(True, alpha=0.3)
    
    ax[1].bar(names, rmses, color=['blue', 'red', 'green'])
    ax[1].set_title('RMSE (ц/га)')
    ax[1].grid(True, alpha=0.3)
    
    ax[2].bar(names, r2s, color=['blue', 'red', 'green'])
    ax[2].set_title('R²')
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100)
    plt.show()
    
    print("\n" + "="*60)
    print("ПРОЕКТ ВЫПОЛНЕН УСПЕШНО!")
    print("="*60)
    print("\nРезультаты сохранены:")
    print("- results_CNN-LSTM.png - графики для лучшей модели")
    print("- model_comparison.png - сравнение всех моделей")

if __name__ == "__main__":
    main()
