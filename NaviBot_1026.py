import logging
import os
import pickle
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import pyodbc
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema import Document as LangDocument
import flask
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# 웹 스크래핑 함수
def scrape_naver(query):
    try:
        logging.debug("Starting scraping for query: %s", query)
        # WebDriver Manager로 ChromeDriver 자동 설치 및 설정
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        
        driver.get("https://map.kakao.com/")
        time.sleep(3)  # 페이지 로드 시간 대기

        # 검색창 찾기 및 입력
        try:
            search_box = driver.find_element(By.CSS_SELECTOR, 'input#search\.keyword\.query')
        except Exception as e:
            logging.error("Search box not found: %s", str(e))
            driver.quit()
            return []
        
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.ENTER)  # 엔터 키를 사용해 검색 실행

        # 결과를 가져오는 중이라는 메시지를 사용자에게 표시
        logging.debug("Fetching results... Please wait.")

        time.sleep(5)  # 검색 결과 로드 대기

        # 상세보기 링크 엘리먼트 찾기
        try:
            details_elements = driver.find_elements(By.CSS_SELECTOR, 'ul#info\.search\.place\.list li .moreview')
        except Exception as e:
            logging.error("Details elements not found: %s", str(e))
            driver.quit()
            return []

        stores = []
        actions = ActionChains(driver)
        for idx, element in enumerate(details_elements, start=1):
            try:
                # 상세보기 클릭 시 추가 정보 수집
                detail_url = element.get_attribute('href')
                driver.execute_script("window.open(arguments[0], '_blank');", detail_url)
                driver.switch_to.window(driver.window_handles[-1])
                time.sleep(2)  # 클릭 후 페이지 로드 대기

                # 후기 탭으로 이동
                try:
                    review_tab = driver.find_element(By.CSS_SELECTOR, 'a.link_evaluation')
                    review_tab.click()
                    time.sleep(2)
                except Exception as e:
                    logging.error("Review tab not found: %s", str(e))
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(2)
                    continue
                
                # 후기 더보기 클릭하여 모든 리뷰 로드 (최대 2번)
                for _ in range(2):
                    try:
                        actions.scroll_by_amount(0, 100).perform()
                        time.sleep(0.5)  # 각 스크롤 사이에 지연 추가
                        more_reviews = driver.find_element(By.CSS_SELECTOR, 'a.link_more')
                        more_reviews.click()
                        time.sleep(2)
                    except Exception:
                        break

                # BeautifulSoup으로 현재 페이지 파싱
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # 가게 정보 수집
                name_tag = soup.find('span', class_='tit_name')
                name = name_tag.text if name_tag else 'Unknown'

                location_tag = soup.find('span', class_='addr')
                location = location_tag.text if location_tag else 'Unknown'

                review_tag = soup.find('em', class_='num_rate')
                review_count = review_tag.text if review_tag else 'Unknown'

                # 리뷰 정보 수집
                reviews = []
                review_elements = soup.find_all('p', class_='txt_comment')
                for review_element in review_elements:
                    review_text = review_element.get_text(strip=True)
                    reviews.append(review_text)

                # 각 가게의 리뷰 출력 로그 추가
                logging.debug(f"Store Name: {name}, Reviews Collected: {reviews}")

                stores.append({
                    'name': name,
                    'location': location,
                    'review_count': review_count,
                    'reviews': reviews
                })

                # 현재 탭 닫기 및 원래 탭으로 돌아가기
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(2)  # 이전 페이지 로드 대기
            except Exception as e:
                logging.error("Error clicking detail link or fetching store info: %s", str(e))

        driver.quit()
        logging.debug("Scraping completed successfully.")
        return stores
    except Exception as e:
        logging.error("Error during scraping: %s", str(e))
        raise

# 감정 분석 함수 (분할 요청 적용)
def analyze_sentiment(reviews):
    try:
        key = "d82a26dfe9944680b3fb8c7819521806"
        endpoint = "https://sentiment-scout.cognitiveservices.azure.com/"
        credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

        positive_keywords = []
        negative_keywords = []

        # 리뷰를 10개 이하로 나누어 요청
        for i in range(0, len(reviews), 10):
            batch = reviews[i:i + 10]
            response = text_analytics_client.analyze_sentiment(documents=batch)

            for review, sentiment_result in zip(batch, response):
                if sentiment_result.sentiment == 'positive':
                    positive_keywords.append(review)
                elif sentiment_result.sentiment == 'negative':
                    negative_keywords.append(review)

        logging.debug("Sentiment analysis completed successfully.")
        return positive_keywords, negative_keywords
    except Exception as e:
        logging.error("Error during sentiment analysis: %s", str(e))
        raise

# 데이터베이스 저장 함수
def save_to_database(stores, search_id, query):
    try:
        server = 'sentimentscout-review.database.windows.net'
        database = 'ReviewDB'
        username = 'testdb'
        password = 'Tiger123'
        driver = '{ODBC Driver 17 for SQL Server}'

        cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password, timeout=30)
        cursor = cnxn.cursor()

        # 테이블 존재 여부 검사 및 생성
        cursor.execute('''
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='StoreInfo' AND xtype='U')
            CREATE TABLE StoreInfo (
                Id INT PRIMARY KEY IDENTITY(1,1),
                SearchId INT,
                Location NVARCHAR(255),
                Place NVARCHAR(255),
                StoreName NVARCHAR(255),
                ReviewCount NVARCHAR(50),
                PositiveKeywords NVARCHAR(MAX),
                NegativeKeywords NVARCHAR(MAX)
            )
        ''')
        cnxn.commit()

        # 데이터 삽입
        for store in stores:
            positive_keywords, negative_keywords = analyze_sentiment(store['reviews'])
            cursor.execute("INSERT INTO StoreInfo (SearchId, Location, Place, StoreName, ReviewCount, PositiveKeywords, NegativeKeywords) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                           (search_id, store['location'], query, store['name'], store['review_count'], ", ".join(positive_keywords), ", ".join(negative_keywords)))
            cnxn.commit()

        cursor.close()
        cnxn.close()
        logging.debug("Database save completed successfully.")
    except pyodbc.Error as e:
        logging.error("Database error: %s", str(e))
        raise

# 챗봇 모델 학습
def get_next_search_id():
    try:
        server = 'sentimentscout-review.database.windows.net'
        database = 'ReviewDB'
        username = 'testdb'
        password = 'Tiger123'
        driver = '{ODBC Driver 17 for SQL Server}'

        cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password, timeout=30)
        cursor = cnxn.cursor()

        cursor.execute("SELECT ISNULL(MAX(SearchId), 0) + 1 FROM StoreInfo")
        row = cursor.fetchone()
        search_id = row[0]

        cursor.close()
        cnxn.close()
        return search_id
    except Exception as e:
        logging.error("Error during fetching next SearchId: %s", str(e))
        raise

# 챗봇 모델 학습 함수
def train_chatbot_model():
    try:
        server = 'sentimentscout-review.database.windows.net'
        database = 'ReviewDB'
        username = 'testdb'
        password = 'Tiger123'
        driver = '{ODBC Driver 17 for SQL Server}'

        cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password, timeout=30)
        cursor = cnxn.cursor()

        cursor.execute("SELECT StoreName, PositiveKeywords, NegativeKeywords FROM StoreInfo")
        rows = cursor.fetchall()

        documents = [LangDocument(page_content=f"Store: {row.StoreName}, Positive: {row.PositiveKeywords}, Negative: {row.NegativeKeywords}") for row in rows]

        embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        db = FAISS.from_documents(documents, embedding_function)

        # 모델 저장
        model_filename = "chatbot_model.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump(db, model_file)
        logging.debug("Model saved successfully to %s", model_filename)

        cursor.close()
        cnxn.close()
        return db
    except Exception as e:
        logging.error("Error during model training: %s", str(e))
        raise

# 챗봇 질문 처리 함수
def process_query_with_model(quest, db):
    try:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5, 'fetch_k': 10})
        context_docs = retriever.get_relevant_documents(quest)

        # 질문 유형 자동 분류 및 적절한 응답 생성
        prompt_template = """
        너는 사용자가 찾는 특정 장소 정보를 기반으로 추천을 제공하는 챗봇이야. 
        반드시 모든 대답은 한글로 해주세요. 
        사용자의 질문에 맞춰 관련 데이터를 제공하세요.
        추천에 대한 답변을 줄 때는 중복된 데이터를 제거하고 알려줘.
        추천할 때는 평점과 리뷰를 종합하여 가장 좋은 장소를 추천해줘.
        단계별로 차근차근 생각하고 알려주세요.
        질문이 빈약해도 로케이션이랑 플레이스를 토대로 검색할 때 어떤 키워드를 많이 사용하는지 알려주고 추천하는 답변을 줘.
        
        질문: {question}
        문서: {context}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatOllama(model="gemma2:9b", temperature=0, base_url="http://127.0.0.1:11434/")
        
        chain = RunnableMap({
            "context": lambda x: context_docs,
            "question": lambda x: quest
        }) | prompt | llm
        
        response = chain.invoke({'question': quest}).content
        return response
    except Exception as e:
        logging.error("Error during query processing: %s", str(e))
        raise

# Flask 기반 챗봇
app = Flask(__name__)

# 웹 인터페이스 추가
@app.route('/chat_ui', methods=['GET', 'POST'])
def chat_ui():
    if request.method == 'POST':
        location = request.form.get('location')
        place = request.form.get('place')
        user_question = request.form.get('question')

        if location and place and user_question:
            full_query = f"{location} {place}"
            try:
                stores = scrape_naver(full_query)
                if not stores:
                    logging.warning("No stores found during scraping. Skipping database save.")
                    return render_template('index.html', error="No stores found during scraping.")
            except Exception as e:
                logging.error("Scraping error: %s", e)
                return render_template('index.html', error="Scraping error occurred.")

            try:
                search_id = get_next_search_id()
                save_to_database(stores, search_id, full_query)
            except Exception as e:
                logging.error("Database error: %s", e)
                return render_template('index.html', error="Database saving error occurred.")

            try:
                db = train_chatbot_model()
                chatbot_response = process_query_with_model(user_question, db)
                return render_template('index.html', response=chatbot_response)
            except Exception as e:
                logging.error("Chatbot error: %s", e)
                return render_template('index.html', error="Chatbot error occurred.")
    return render_template('index.html')

# 기본 경로 정의
@app.route('/')
def home():
    return render_template('index.html')

# 챗봇 질문 처리 엔드포인트
@app.route('/chat', methods=['POST'])
def chat():
    try:
        location = request.form.get('location', '서울 강남')  # 기본값 설정
        place = request.form.get('place', '치과')  # 기본값 설정
        user_input = request.form.get('question')
        
        stores = scrape_naver(f"{location} {place}")
        search_id = get_next_search_id()  # Get the next available SearchId value
        query = f"{location} {place}"
        save_to_database(stores, search_id, query)

        db = train_chatbot_model()
        response = process_query_with_model(user_input, db)

        return jsonify({"response": response})
    except Exception as e:
        logging.error("Unhandled error in chat endpoint: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
