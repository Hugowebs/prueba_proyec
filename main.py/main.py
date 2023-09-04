from fastapi import FastAPI
import pandas as pd


app = FastAPI()

@app.get('/userdata/{user_id}')
def userdata(user_id: str):
    user_data = pd.read_csv('user_data.csv')
    user = user_data[user_data['user_id'] == user_id]
    if user.empty:
        return {"message": "Usuario no encontrado"}
    
    money_spent = user['price'].sum()
    recommendation = user['recommend'].mean()
    items_count = user['items_count'].values[0]
    
    return {"money_spent": money_spent, "recommendation": recommendation, "items_count": items_count}

@app.get('/countreviews/{start_date}/{end_date}')
def countreviews(start_date: str, end_date: str):
    count_reviews = pd.read_csv('count_reviews.csv')
    reviews_between_dates = count_reviews[(count_reviews['posted'] >= start_date) & (count_reviews['posted'] <= end_date)]
    
    if reviews_between_dates.empty:
        return {"message": "No se encontraron reseñas en las fechas especificadas"}
    
    user_count = len(reviews_between_dates)
    recommendation_percentage = (reviews_between_dates['recommend'].sum() / user_count) * 100
    
    return {"user_count": user_count, "recommendation_percentage": recommendation_percentage}

@app.get('/genero/{genero}')
def genero(genero: str):
    genre_data = pd.read_csv('genre_data.csv')
    genre_ranking = genre_data.groupby([f'genre{i}' for i in range(3)]).mean().sort_values(by='PlayTimeForever', ascending=False)
    genre_rank = genre_ranking.index.get_loc(genero)
    
    return {"genre_rank": genre_rank}

@app.get('/userforgenre/{genero}')
def userforgenre(genero: str):
    genre_data = pd.read_csv('genre_data.csv')
    top_users = genre_data[genre_data['genre0'] == genero].nlargest(5, 'playtime_forever')
    top_users = top_users[['user_id', 'URL', 'playtime_forever']].to_dict(orient='records')
    
    return top_users

@app.get('/desarrollador/{desarrollador}')
def desarrollador(desarrollador: str):
    developer = pd.read_csv('developer_data.csv')
    developer_info = developer[developer['developer'] == desarrollador]
    if developer_info.empty:
        return {"message": "Desarrollador no encontrado"}
    
    items_count_by_year = developer_info.groupby('posted').agg({'items_count': 'sum'}).to_dict(orient='index')
    free_content_percentage_by_year = developer_info.groupby('posted').agg({'free_content_percentage': 'mean'}).to_dict(orient='index')
    
    return {"items_count_by_year": items_count_by_year, "free_content_percentage_by_year": free_content_percentage_by_year}

@app.get('/sentiment_analysis/{year}')
def sentiment_analysis(year: int):
    sentiment_analysis = pd.read_csv('func_sentiment_data.csv')
    sentiment_data = sentiment_analysis[sentiment_analysis['posted'].dt.year == year]
    sentiment_counts = sentiment_data['sentiment_analysis'].value_counts().to_dict()
    
    return sentiment_counts