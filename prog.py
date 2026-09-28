import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os

def main():
    print("1. Загрузка данных...")
    queries_df = pd.read_parquet('benchmark_queries.parquet')
    items_df = pd.read_parquet('benchmark_items.parquet')

    for col in ['item_title_raw', 'item_infm_params_text', 'item_description_raw']:
        items_df[col] = items_df[col].fillna('')
    for col in ['search_query', 'search_infm_params_text']:
        queries_df[col] = queries_df[col].fillna('')

    print("2. Умная подготовка текстов (Хак с весами)...")
    # Умножение заголовкадля повышения важности
    items_text = (
        items_df['item_title_raw'] + " " + 
        items_df['item_title_raw'] + " " + 
        items_df['item_title_raw'] + " " + 
        items_df['item_infm_params_text'] + " " + 
        items_df['item_description_raw']
    ).str.lower()
    
    queries_text = (
        queries_df['search_query'] + " " + 
        queries_df['search_infm_params_text']
    ).str.lower()

    print("3. Продвинутая векторизация (BM25-like)...")
    # sublinear_tf=True делает рост веса слова логарифмическим (как в BM25)
    # min_df=2 убирает слова с опечатками, которые встречаются 1 раз
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=150000, 
        sublinear_tf=True, 
        min_df=2 
    )
    
    items_matrix = vectorizer.fit_transform(items_text)
    queries_matrix = vectorizer.transform(queries_text)

    print("4. Быстрый поиск топ-50 кандидатов...")
    predictions = []
    batch_size = 500
    item_ids = items_df['item_id'].values

    for i in tqdm(range(0, queries_matrix.shape[0], batch_size)):
        batch_queries = queries_matrix[i : i + batch_size]
        
        # Считаем похожесть
        similarity_scores = batch_queries.dot(items_matrix.T).toarray()
        
        for row in similarity_scores:
            # Используем быструю частичную сортировку
            top50_idx_unsorted = np.argpartition(row, -50)[-50:]
            top50_idx = top50_idx_unsorted[np.argsort(row[top50_idx_unsorted])[::-1]]
            predictions.append(item_ids[top50_idx])

    print("5. Сохранение результата...")
    answer = pd.DataFrame({
        'query_id': queries_df['query_id'],
        'answer': [' '.join(top50) for top50 in predictions]
    })
    
    answer.to_csv('answer.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()