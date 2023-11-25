from fastapi import FastAPI
from typing import List
from datetime import datetime
from schema import PostGet
import os

from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine

app = FastAPI()


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path('/home/user/...')
    local_model = CatBoostClassifier()
    local_model.load_model(model_path)  # пример как можно загружать модели
    return local_model


def batch_load_user(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://..."
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql_query(query, conn, chunksize=CHUNKSIZE, dtype={"gender": "category",
       "age": "uint8", "country": "category"}):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def batch_load_post(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://..."
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql_query(query, conn, chunksize=CHUNKSIZE, dtype={"city": "category", "exp_group": "uint8", "os": "category", "source": "category", "topic": "category", "user_views": 'uint8'}):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def batch_load_sql(query: str) -> pd.DataFrame:
    """Функция для оптимизации выгрузки данных из БД по памяти.
        Выгружает данные частями."""
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://..."
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_all_posts() -> pd.DataFrame:
    query = f"""SELECT * 
             FROM public.post_text_df"""
    local_df = batch_load_sql(query)
    return local_df


def load_user_features() -> pd.DataFrame:
    query = """SELECT timestamp, user_id, post_id, gender,
       age, country
                FROM public.acylhan_lesson_22_7"""
    df = batch_load_user(query)
    return df


def load_post_features() -> pd.DataFrame:
    query = """SELECT city, exp_group, os, source, topic,
       tfidf_mean, user_views, post_views
                FROM public.acylhan_lesson_22_7"""
    df = batch_load_post(query)
    return df


model = load_models()

user_df = load_user_features()
post_df = load_post_features()
posts = load_all_posts()
global_df = pd.concat([user_df, post_df], axis=1)


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5):

    user = global_df[global_df['user_id'] == id]

    user['pred'] = model.predict(user.drop(['user_id', 'post_id'], axis=1))
    
    pred_list = list(user[user['pred'] == 1]['post_id'].values)

    if len(pred_list) < limit:
        pred_list = pred_list + list(user[user['pred'] == 0].iloc[:limit - len(pred_list) + 1]['post_id'].values)
        
    top_post = posts[posts['post_id'].isin(pred_list)]
    rec_list = []
    
    for i in range(top_post.shape[0]):
        post = top_post.iloc[i]
        rec_list.append(PostGet(id=post['post_id'], text=post['text'], topic=post['topic']))

    return rec_list


