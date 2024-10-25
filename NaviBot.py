from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pyodbc
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from selenium.webdriver.common.by import By
from flask import Flask, request, jsonify

# 웹 스크래핑
def scrape_naver(query):
    driver = webdriver.Chrome(executable_path='chromedriver_path')  # chromedriver 경로 지정
    driver.get("https://www.naver.com")
    
    search_box = driver.find_element(By.NAME, 'query')  # 최신 방식
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    
    time.sleep(3)  # 페이지 로드 시간 대기
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    stores = []
    store_elements = soup.find_all('div', class_='place_detail')  # 상점 정보 포함된 태그
    
    for store in store_elements:
        name = store.find('span', class_='name').text
        rating = store.find('span', class_='rating').text
        reviews = store.find_all('span', class_='review')
        
        store_reviews = [review.text for review in reviews]
        
        stores.append({
            'name': name,
            'rating': rating,
            'reviews': store_reviews
        })
    
    driver.quit()
    return stores

#DB
def save_to_database(stores):
    server = 'sentimentscout-review.database.windows.net'
    database = 'ReviewDB'
    username = 'testdb'
    password = 'Tiger123'
    driver = '{ODBC Driver 17 for SQL Server}'
    
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+password)
    cursor = cnxn.cursor()

    # 테이블 존재 여부 검사 및 생성
    cursor.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='StoreInfo' AND xtype='U')
        CREATE TABLE StoreInfo (
            StoreName NVARCHAR(100),
            Rating FLOAT,
            Review NVARCHAR(MAX)
        )
    ''')
    cnxn.commit()

    # 데이터 삽입
    for store in stores:
        name = store['name']
        rating = store['rating']
        for review in store['reviews']:
            cursor.execute("INSERT INTO StoreInfo (StoreName, Rating, Review) VALUES (?, ?, ?)", (name, rating, review))
            cnxn.commit()
    
    cursor.close()
    cnxn.close()


# Step 3: 감정 분석 함수
def analyze_sentiment(reviews):
    key = "d82a26dfe9944680b3fb8c7819521806"
    endpoint = "https://sentiment-scout.cognitiveservices.azure.com/"
    credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

    response = text_analytics_client.analyze_sentiment(documents=reviews)

    positive_reviews = sum(1 for r in response if r.sentiment == 'positive')
    negative_reviews = sum(1 for r in response if r.sentiment == 'negative')
    for idx, doc in enumerate(response):
        print(f"문서 {idx+1}: 감정 {doc.sentiment}, 긍정 점수: {doc.confidence_scores.positive}, 부정 점수: {doc.confidence_scores.negative}")
    return positive_reviews, negative_reviews


# Step 4: Flask 기반 챗봇
app = Flask(__name__)
# 기본 경로 정의
@app.route('/')
def home():
    return "Flask 서버가 정상적으로 실행 중입니다. /chat 경로로 POST 요청을 보내주세요."
@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
        user_input = request.json.get('message')  # 사용자의 입력
        if "치과" in user_input and "추천" in user_input:
            # 스크래핑 수행
            stores = scrape_naver("서울 강남 치과")
            save_to_database(stores)
            
            # 감정 분석 및 추천 수행
            for store in stores:
                reviews = store['reviews']
                positive_reviews, negative_reviews = analyze_sentiment(reviews)
                total_reviews = len(reviews)
                positive_ratio = (positive_reviews / total_reviews) * 100 if total_reviews > 0 else 0
                
                store['positive_ratio'] = positive_ratio

            # 가장 긍정적인 치과 추천
            best_store = max(stores, key=lambda x: x['positive_ratio'])
            response_text = f"{best_store['name']} 치과는 평점 {best_store['rating']}점이며, 긍정적인 리뷰 비율이 {best_store['positive_ratio']}%입니다."
            return jsonify({"response": response_text})
        else:
            return jsonify({"response": "질문을 이해하지 못했습니다. 다시 입력해 주세요."})
    return jsonify({"error": "Invalid request, JSON expected"}), 400

if __name__ == '__main__':
    app.run(debug=True)
