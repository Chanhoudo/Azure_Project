import requests

# Flask 서버에 보낼 URL
url = 'http://127.0.0.1:5000/chat'

# 서버에 전송할 데이터 (메시지 내용)
data = {'message': '서울 강남 치과 추천해줘'}

# POST 요청을 보내고 응답 받기
response = requests.post(url, json=data)

# 응답을 출력
print(response.json())
