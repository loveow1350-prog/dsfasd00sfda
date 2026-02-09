# Redis 서버 설치 및 실행 가이드

## 중요: Redis는 선택사항입니다!

**Redis 서버 없이도 프로그램은 정상 작동합니다.** Redis는 캐싱을 통한 성능 최적화용이며, 없으면 캐싱만 비활성화됩니다.

## 방법 1: WSL (Ubuntu) 사용 (추천)

WSL이 이미 설치되어 있으므로 이 방법이 가장 간단합니다.

### 1단계: WSL 터미널 열기
```bash
wsl
```

### 2단계: Redis 설치
```bash
# 패키지 목록 업데이트
sudo apt-get update

# Redis 서버 설치
sudo apt-get install redis-server -y
```

### 3단계: Redis 서버 시작
```bash
# Redis 서버 시작
sudo service redis-server start

# 상태 확인
sudo service redis-server status
```

### 4단계: 연결 테스트
```bash
# Redis CLI로 테스트
redis-cli ping
# 응답: PONG 이면 성공!
```

### 5단계: 자동 시작 설정 (선택사항)
```bash
# WSL 시작 시 자동으로 Redis 실행
echo "sudo service redis-server start" >> ~/.bashrc
```

### Redis 서버 제어 명령
```bash
# 시작
sudo service redis-server start

# 중지
sudo service redis-server stop

# 재시작
sudo service redis-server restart

# 상태 확인
sudo service redis-server status
```

## 방법 2: Windows용 Redis (대안)

### 다운로드 및 설치
1. https://github.com/microsoftarchive/redis/releases
2. `Redis-x64-3.0.504.msi` 다운로드 및 설치
3. 설치 후 자동으로 서비스로 실행됨

### Redis 서비스 관리 (Windows)
```powershell
# 서비스 상태 확인
Get-Service redis

# 서비스 시작
Start-Service redis

# 서비스 중지
Stop-Service redis

# 서비스 재시작
Restart-Service redis
```

## 방법 3: Docker 사용

```bash
# Redis 컨테이너 실행
docker run -d -p 6379:6379 --name redis redis:latest

# 컨테이너 중지
docker stop redis

# 컨테이너 시작
docker start redis
```

## Python에서 Redis 연결 확인

```python
import redis

try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("✅ Redis 서버 연결 성공!")
except redis.ConnectionError:
    print("❌ Redis 서버에 연결할 수 없습니다.")
    print("   프로그램은 캐싱 없이 계속 실행됩니다.")
```

## 프로젝트에서 Redis 연결 테스트

```bash
cd /mnt/c/Users/pegoo/Desktop/nlp_project_2

python -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print('✅ Redis 서버 연결 성공!')
except Exception as e:
    print(f'❌ Redis 연결 실패: {e}')
    print('   프로그램은 캐싱 없이 계속 실행됩니다.')
"
```

## 빠른 시작 (WSL)

```bash
# 1. WSL 열기
wsl

# 2. Redis 설치 (한 번만)
sudo apt-get update && sudo apt-get install redis-server -y

# 3. Redis 시작
sudo service redis-server start

# 4. 테스트
redis-cli ping

# 5. Windows PowerShell로 돌아가기
exit

# 6. 프로젝트 실행
cd C:\Users\pegoo\Desktop\nlp_project_2
python main_pipeline.py sample/중간보고서_자연어처리.pdf
```

## 문제 해결

### "redis-cli: command not found"
```bash
sudo apt-get install redis-tools -y
```

### "Could not connect to Redis at 127.0.0.1:6379"
```bash
# Redis 서버가 실행 중인지 확인
sudo service redis-server status

# 실행 중이 아니면 시작
sudo service redis-server start
```

### 포트 6379가 이미 사용 중
```bash
# 실행 중인 Redis 프로세스 확인
ps aux | grep redis

# 필요시 중지 후 재시작
sudo service redis-server restart
```

## Redis 없이 실행하기

Redis 서버를 설치/실행하지 않아도 프로그램은 정상 작동합니다:

```bash
# Redis 없이 바로 실행
python main_pipeline.py sample/중간보고서_자연어처리.pdf

# 다음과 같은 경고 메시지가 나타날 수 있지만 무시해도 됩니다:
# WARNING: Could not connect to Redis. Caching disabled.
```
## 참고

- Redis 서버는 포트 6379에서 실행됩니다.
- WSL의 Redis는 Windows에서도 `localhost:6379`로 접근 가능합니다.
- 캐싱은 LLM 호출과 검색 결과를 저장하여 반복 실행 시 속도를 높입니다.