import pandas as pd
import joblib  # モデルを読み込むためにjoblibをインポート

class ScoringService(object):

    @classmethod
    def expand_datetime(cls,df):
        if 'datetime' in df.columns:
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofyear'] = df['datetime'].dt.dayofyear
            df['weekofyear'] = df['datetime'].dt.weekofyear
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['hour'] = df['datetime'].dt.hour

        return df
    

    @classmethod
    def get_model(cls, model_path, inference_df, inference_log):
        """モデルと推論補助データを読み込む

        Args:
            model_path (str): 学習済みモデルが格納されているディレクトリのパス
            inference_df (DataFrame): 推論対象の期間よりも過去のDataFrame
            inference_log (DataFrame): 使用されていないが、将来的な拡張のために保持

        Returns:
            bool: モデルの読み込みに成功したかどうか
        """
        try:
            # 指定されたパスからモデルをロード
            cls.model = joblib.load(f"{model_path}/model.joblib")
            cls.data = inference_df  # この例では使用されていませんが、将来的な使用のために保持
            cls.log_pathes = inference_log  # この例では使用されていませんが、将来的な使用のために保持
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False

    @classmethod
    def predict(cls, input, input_log):
        """入力データに対する予測を行う

        Args:
            input (DataFrame): 当日のデータ
            input_log (DataFrame): 使用されていないが、将来的な拡張のために保持

        Returns:
            DataFrame: 予測結果を含むDataFrame
        """
        features = input[['datetime', 'KP', 'OCC', 'allCars', 'speed', 'is_congestion',
       'road_code', 'limit_speed', 'start_KP', 'end_KP', 'start_pref_code',
       'end_pref_code', 'start_degree', 'end_degree', 'search_specified',
       'search_unspecified','start_code', 'end_code','direction']]
        
        features['datetime'] = pd.to_datetime(features['datetime'])
        
        replace_dict = {'下り': 0, '上り': 1} # この辞書を編集して任意の変換ルールを定義
        features['direction'] = features['direction'].replace(replace_dict)
        features['point'] = features['start_code'].astype(str) + "_" + features['direction'].astype(str) + "_" + features['end_code'].astype(str)
        features = cls.expand_datetime(features)

        features.drop(columns=['datetime', 'start_code', 'end_code','direction'], inplace=True)
            # モデルを使用して予測を実行
        predictions = cls.model.predict(features)

        # 予測結果をDataFrameに追加（予測結果がNumPy配列の場合）
        results_df = input.copy()
        results_df['prediction'] = predictions

        
        results_df=results_df[['datetime', 'start_code', 'end_code', 'KP', 'prediction']]

        return results_df