# ADP(데이터분석 전문가) 시험 전체 과정 가이드

## 📋 목차
1. [ADP 시험 개요](#adp-시험-개요)
2. [전체 학습 흐름도](#전체-학습-흐름도)
3. [과목별 상세 내용](#과목별-상세-내용)
4. [학습 로드맵](#학습-로드맵)
5. [실전 준비 전략](#실전-준비-전략)

---

## ADP 시험 개요

### 시험 구조

| 구분 | 과목명 | 문항수 | 배점 | 시간 |
|------|--------|--------|------|------|
| 1과목 | 데이터 이해 | 20문항 | 20점 | 90분 |
| 2과목 | 데이터 분석 기획 | 20문항 | 20점 | (1~3과목 통합) |
| 3과목 | 데이터 분석 | 50문항 | 60점 | |
| **합계** | | **90문항** | **100점** | **90분** |

### 합격 기준
- **총점 75점 이상** (과목별 과락 없음)
- 객관식 4지선다형
- CBT(Computer Based Test) 방식

---

## 전체 학습 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│                    ADP 시험 전체 과정                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   [1과목: 20점]       [2과목: 20점]       [3과목: 60점]
   데이터 이해         데이터 분석 기획     데이터 분석
        ↓                   ↓                   ↓
    ┌───────┐          ┌───────┐          ┌───────────┐
    │Ch1~2  │          │Ch1~2  │          │Ch1~5      │
    │이론   │          │기획   │          │분석+통계  │
    │기초   │          │설계   │          │+머신러닝  │
    └───────┘          └───────┘          └───────────┘
        ↓                   ↓                   ↓
    [30% 비중]         [30% 비중]         [40% 비중]
        └───────────────────┼───────────────────┘
                            ↓
                    [실전 모의고사]
                            ↓
                    [최종 점검 & 시험]
```

---

## 과목별 상세 내용

## 1과목: 데이터 이해 (20점)

### 📊 전체 구조

| 장 | 주제 | 세부 내용 | 중요도 | 문항수(예상) |
|---|------|-----------|--------|-------------|
| 1장 | 데이터의 이해 | 데이터, 정보, 지식 개념<br>데이터베이스 개요<br>빅데이터 개념 및 활용 | ⭐⭐⭐ | 8~10 |
| 2장 | 데이터베이스 | 데이터베이스 정의 및 특징<br>데이터베이스 활용 | ⭐⭐⭐⭐ | 10~12 |

---

### 1장: 데이터의 이해

#### 1.1 데이터, 정보, 지식의 관계

| 개념 | 정의 | 특징 | 예시 |
|------|------|------|------|
| **데이터(Data)** | 가공되지 않은 순수한 사실, 값 | - 객관적 사실<br>- 가공 전 상태<br>- 의미 없는 기호/숫자 | 15, 20, 30 (온도 측정값) |
| **정보(Information)** | 데이터를 가공하여 의미를 부여한 것 | - 처리/정리된 데이터<br>- 특정 목적에 맞게 가공<br>- 의사결정에 활용 | "오늘 기온은 평년보다 5도 높음" |
| **지식(Knowledge)** | 정보를 체계화하고 내재화한 것 | - 경험과 학습을 통한 이해<br>- 패턴 인식<br>- 예측 및 판단 가능 | "이 지역은 여름철 평균 기온이 상승 추세" |

#### 1.2 데이터의 유형

| 구분 기준 | 유형 | 설명 | 예시 |
|-----------|------|------|------|
| **구조화 정도** | 정형 데이터 | - 고정된 스키마<br>- RDBMS에 저장<br>- 행/열 구조 | 관계형 DB 테이블, Excel |
| | 반정형 데이터 | - 스키마와 데이터 혼재<br>- 메타데이터 포함 | XML, JSON, 웹 로그 |
| | 비정형 데이터 | - 정해진 구조 없음<br>- 텍스트, 멀티미디어 | 이메일, 영상, 음성, SNS |
| **저장 위치** | 내부 데이터 | 조직 내부에서 생성/관리 | 거래 데이터, 고객 정보 |
| | 외부 데이터 | 외부에서 수집한 데이터 | 공공 데이터, 소셜 미디어 |

#### 1.3 빅데이터의 이해

##### 빅데이터 3V (확장 5V)

| V | 개념 | 설명 | 특징 |
|---|------|------|------|
| **Volume** | 데이터 크기 | 테라바이트(TB) ~ 페타바이트(PB) 규모 | 대용량 저장/처리 기술 필요 |
| **Velocity** | 데이터 속도 | 실시간 생성 및 처리 | 스트리밍 처리, 실시간 분석 |
| **Variety** | 데이터 다양성 | 정형/반정형/비정형 다양한 형태 | 멀티 소스 통합 필요 |
| **Veracity** | 데이터 정확성 | 데이터의 신뢰성과 품질 | 데이터 검증 및 정제 중요 |
| **Value** | 데이터 가치 | 분석을 통한 가치 창출 | 비즈니스 인사이트 도출 |

##### 빅데이터 처리 기술

| 기술 | 설명 | 대표 기술 |
|------|------|-----------|
| **분산 저장** | 여러 서버에 데이터 분산 저장 | HDFS, NoSQL(MongoDB, Cassandra) |
| **분산 처리** | 병렬 처리로 빠른 분석 | Hadoop MapReduce, Spark |
| **실시간 처리** | 스트리밍 데이터 실시간 처리 | Kafka, Flink, Storm |
| **분석 플랫폼** | 통합 분석 환경 | Databricks, Cloudera |

---

### 2장: 데이터베이스

#### 2.1 데이터베이스 기본 개념

| 구분 | 내용 |
|------|------|
| **정의** | 특정 조직의 여러 사용자가 공유하여 사용할 수 있도록 통합·저장된 운영 데이터의 집합 |
| **특징** | - 실시간 접근성(Real-time Accessibility)<br>- 계속적인 변화(Continuous Evolution)<br>- 동시 공유(Concurrent Sharing)<br>- 내용에 의한 참조(Content Reference) |

#### 2.2 데이터베이스 관리 시스템 (DBMS)

##### DBMS의 주요 기능

| 기능 | 설명 | 세부 내용 |
|------|------|-----------|
| **정의 기능** | 데이터 구조 정의 | DDL(Data Definition Language) 사용 |
| **조작 기능** | 데이터 검색, 삽입, 수정, 삭제 | DML(Data Manipulation Language) 사용 |
| **제어 기능** | 데이터 무결성, 보안, 동시성 제어 | DCL(Data Control Language) 사용 |

##### DBMS 유형

| 유형 | 특징 | 장점 | 단점 | 예시 |
|------|------|------|------|------|
| **계층형 DB** | 트리 구조<br>1:N 관계 | - 빠른 접근<br>- 명확한 구조 | - 유연성 부족<br>- 복잡한 관계 표현 어려움 | IMS |
| **네트워크형 DB** | 그래프 구조<br>M:N 관계 | - 복잡한 관계 표현<br>- 빠른 탐색 | - 구조 복잡<br>- 유지보수 어려움 | IDMS |
| **관계형 DB (RDBMS)** | 테이블 구조<br>SQL 사용 | - 데이터 독립성<br>- 유연성<br>- 표준화된 질의어 | - 대용량 처리 한계<br>- 복잡한 조인 비용 | Oracle, MySQL, PostgreSQL |
| **객체지향 DB** | 객체 저장<br>상속/캡슐화 | - 복잡한 데이터 모델링<br>- 재사용성 | - 표준 부족<br>- 성능 이슈 | ObjectDB |
| **NoSQL** | 비정형 데이터<br>분산 저장 | - 수평 확장성<br>- 유연한 스키마<br>- 빠른 처리 | - 트랜잭션 보장 약함<br>- 표준화 부족 | MongoDB, Redis, Cassandra |

#### 2.3 NoSQL 데이터베이스 상세

| 유형 | 데이터 모델 | 특징 | 사용 사례 | 대표 제품 |
|------|------------|------|-----------|-----------|
| **Key-Value** | 키-값 쌍 | - 가장 단순한 구조<br>- 빠른 조회<br>- 메모리 기반 | 캐싱, 세션 관리 | Redis, DynamoDB |
| **Document** | JSON/BSON 문서 | - 스키마 유연<br>- 중첩 구조 지원 | 콘텐츠 관리, 사용자 프로필 | MongoDB, CouchDB |
| **Column-Family** | 컬럼 기반 저장 | - 대용량 쓰기 최적화<br>- 압축 효율적 | 시계열 데이터, 로그 분석 | Cassandra, HBase |
| **Graph** | 노드와 엣지 | - 관계 중심<br>- 복잡한 연결 쿼리 | 소셜 네트워크, 추천 시스템 | Neo4j, JanusGraph |

#### 2.4 SQL 기초

##### DDL (Data Definition Language)

| 명령어 | 기능 | 예시 |
|--------|------|------|
| CREATE | 객체 생성 | `CREATE TABLE users (id INT, name VARCHAR(50));` |
| ALTER | 객체 수정 | `ALTER TABLE users ADD email VARCHAR(100);` |
| DROP | 객체 삭제 | `DROP TABLE users;` |
| TRUNCATE | 데이터 전체 삭제 (구조 유지) | `TRUNCATE TABLE users;` |

##### DML (Data Manipulation Language)

| 명령어 | 기능 | 예시 |
|--------|------|------|
| SELECT | 데이터 조회 | `SELECT * FROM users WHERE age > 20;` |
| INSERT | 데이터 삽입 | `INSERT INTO users VALUES (1, 'Hong', 25);` |
| UPDATE | 데이터 수정 | `UPDATE users SET age = 26 WHERE id = 1;` |
| DELETE | 데이터 삭제 | `DELETE FROM users WHERE id = 1;` |

##### DCL (Data Control Language)

| 명령어 | 기능 | 예시 |
|--------|------|------|
| GRANT | 권한 부여 | `GRANT SELECT ON users TO user1;` |
| REVOKE | 권한 회수 | `REVOKE SELECT ON users FROM user1;` |

##### JOIN 유형

| JOIN 유형 | 설명 | 결과 |
|-----------|------|------|
| **INNER JOIN** | 양쪽 테이블에 모두 존재하는 데이터만 | 교집합 |
| **LEFT OUTER JOIN** | 왼쪽 테이블의 모든 데이터 + 매칭되는 오른쪽 데이터 | 왼쪽 기준 |
| **RIGHT OUTER JOIN** | 오른쪽 테이블의 모든 데이터 + 매칭되는 왼쪽 데이터 | 오른쪽 기준 |
| **FULL OUTER JOIN** | 양쪽 테이블의 모든 데이터 | 합집합 |
| **CROSS JOIN** | 두 테이블의 카티션 곱 | A×B |

---

## 2과목: 데이터 분석 기획 (20점)

### 📊 전체 구조

| 장 | 주제 | 세부 내용 | 중요도 | 문항수(예상) |
|---|------|-----------|--------|-------------|
| 1장 | 데이터 분석 기획의 이해 | 분석 방법론<br>분석 마스터 플랜<br>분석 거버넌스 | ⭐⭐⭐⭐⭐ | 12~15 |
| 2장 | 분석 과제 발굴 | 분석 과제 도출<br>분석 프로젝트 관리 | ⭐⭐⭐ | 5~8 |

---

### 1장: 데이터 분석 기획의 이해

#### 1.1 분석 방법론 비교

| 방법론 | 개발사 | 특징 | 주요 단계 | 적용 분야 |
|--------|--------|------|----------|-----------|
| **KDD** | - | 학술적 접근<br>데이터 마이닝 중심 | Selection → Preprocessing → Transformation → Data Mining → Interpretation | 연구, 탐색적 분석 |
| **CRISP-DM** | SPSS | 산업 표준<br>순환적 프로세스 | Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment | 일반 비즈니스 분석 |
| **SEMMA** | SAS | SAS 도구 중심<br>기술적 접근 | Sample → Explore → Modify → Model → Assess | SAS 환경 분석 |
| **빅데이터 분석 방법론** | 한국정보화진흥원 | 빅데이터 특화<br>한국형 방법론 | 분석 기획 → 데이터 준비 → 데이터 분석 → 시스템 구현 → 평가 및 전개 | 빅데이터 프로젝트 |

#### 1.2 CRISP-DM 상세 단계

| 단계 | 핵심 활동 | 주요 산출물 | 소요 시간(%) |
|------|----------|------------|-------------|
| **1. 업무 이해<br>(Business Understanding)** | - 비즈니스 목표 설정<br>- 현황 평가<br>- 분석 목표 정의<br>- 프로젝트 계획 수립 | - 프로젝트 계획서<br>- 초기 리스크 평가서 | 15~20% |
| **2. 데이터 이해<br>(Data Understanding)** | - 초기 데이터 수집<br>- 데이터 기술 분석<br>- 데이터 탐색<br>- 데이터 품질 검증 | - 데이터 수집 보고서<br>- 데이터 품질 보고서 | 20~25% |
| **3. 데이터 준비<br>(Data Preparation)** | - 데이터 선택<br>- 데이터 정제<br>- 데이터 변환<br>- 통합 및 포맷팅 | - 정제된 데이터셋<br>- 파생변수 정의서 | 30~40% |
| **4. 모델링<br>(Modeling)** | - 모델링 기법 선택<br>- 테스트 설계<br>- 모델 구축<br>- 모델 평가 | - 분석 모델<br>- 모델 성능 보고서 | 15~20% |
| **5. 평가<br>(Evaluation)** | - 결과 평가<br>- 프로세스 검토<br>- 다음 단계 결정 | - 평가 보고서<br>- 개선 권고사항 | 5~10% |
| **6. 전개<br>(Deployment)** | - 전개 계획 수립<br>- 모니터링 및 유지보수<br>- 최종 보고서 작성<br>- 프로젝트 검토 | - 배포 계획서<br>- 최종 보고서<br>- 운영 매뉴얼 | 5~10% |

#### 1.3 분석 과제 유형

| 유형 | 목적 | 접근 방법 | 기법 예시 | 활용 사례 |
|------|------|----------|----------|-----------|
| **최적화(Optimization)** | 최적의 의사결정<br>자원 효율화 | - 목적함수 정의<br>- 제약조건 설정<br>- 최적해 탐색 | 선형계획법, 정수계획법, 시뮬레이션 | 물류 경로 최적화, 재고 관리, 인력 배치 |
| **솔루션(Solution)** | 문제 해결<br>개선 방안 도출 | - 문제 정의<br>- 원인 분석<br>- 해결책 제시 | 의사결정나무, 연관분석, 군집분석 | 이탈 원인 분석, 불량 요인 파악 |
| **인사이트(Insight)** | 숨겨진 패턴 발견<br>새로운 관점 제시 | - 탐색적 분석<br>- 패턴 인식<br>- 가설 수립 | 시각화, 기술통계, 상관분석 | 고객 세분화, 트렌드 분석 |
| **예측(Prediction)** | 미래 값 예측<br>리스크 예측 | - 과거 데이터 학습<br>- 패턴 모델링<br>- 미래 추정 | 회귀분석, 시계열, 머신러닝 | 수요 예측, 매출 예측, 신용 평가 |

#### 1.4 분석 거버넌스 체계

##### 데이터 거버넌스 구성 요소

| 구성 요소 | 설명 | 주요 내용 |
|-----------|------|-----------|
| **원칙(Principle)** | 데이터 관리 기본 방향 | - 데이터 표준화<br>- 품질 관리 원칙<br>- 보안 정책 |
| **조직(Organization)** | 역할 및 책임 | - 데이터 관리 위원회<br>- 데이터 스튜어드<br>- 데이터 아키텍트 |
| **프로세스(Process)** | 데이터 관리 절차 | - 데이터 생성/수집<br>- 저장/활용<br>- 폐기 프로세스 |

##### 마스터 데이터 관리(MDM)

| 영역 | 설명 | 관리 대상 |
|------|------|-----------|
| **고객 MDM** | 고객 정보 통합 관리 | 고객 ID, 인적사항, 거래이력 |
| **제품 MDM** | 제품 정보 통합 관리 | 제품 코드, 속성, 분류체계 |
| **공급업체 MDM** | 협력사 정보 관리 | 업체 코드, 계약정보, 평가이력 |

---

### 2장: 분석 과제 발굴

#### 2.1 분석 과제 도출 프레임워크

##### 하향식 접근법 (Top-Down)

| 단계 | 활동 | 산출물 | 특징 |
|------|------|--------|------|
| **1. 전략 분석** | - 경영 전략 이해<br>- 핵심 성공 요인(KSF) 도출 | 전략 맵 | 경영진 주도 |
| **2. 과제 정의** | - 전략 목표별 과제 도출<br>- 우선순위 평가 | 과제 후보 리스트 | 체계적 접근 |
| **3. 실행 계획** | - 로드맵 수립<br>- 자원 배분 계획 | 실행 계획서 | 장기적 관점 |

##### 상향식 접근법 (Bottom-Up)

| 단계 | 활동 | 산출물 | 특징 |
|------|------|--------|------|
| **1. 문제 탐색** | - 현업 인터뷰<br>- Pain Point 수집 | 문제 목록 | 현업 중심 |
| **2. 아이디어 발굴** | - 브레인스토밍<br>- Quick Win 과제 발굴 | 아이디어 카드 | 신속한 실행 |
| **3. 타당성 검토** | - 실현 가능성 평가<br>- 효과 추정 | 과제 평가서 | 실용적 접근 |

#### 2.2 과제 우선순위 평가

| 평가 기준 | 세부 항목 | 점수 | 가중치 |
|-----------|----------|------|--------|
| **전략적 중요도** | - 경영 목표 부합도<br>- 전사적 영향도<br>- 시급성 | 1~5점 | 40% |
| **비즈니스 효과** | - 매출/비용 효과<br>- ROI<br>- 정량적 효과 | 1~5점 | 30% |
| **실행 용이성** | - 데이터 가용성<br>- 기술적 난이도<br>- 소요 기간 | 1~5점 | 20% |
| **리스크** | - 실패 가능성<br>- 부작용<br>- 저항 정도 | 1~5점 | 10% |

#### 2.3 분석 프로젝트 관리

##### 프로젝트 단계별 관리 포인트

| 단계 | 관리 항목 | 주요 활동 | 리스크 관리 |
|------|----------|----------|------------|
| **착수** | 범위, 일정, 자원 | - 킥오프 미팅<br>- 역할 정의<br>- 일정 수립 | - 목표 모호성<br>- 이해관계자 갈등 |
| **수행** | 품질, 진척도 | - 정기 리뷰<br>- 이슈 관리<br>- 변경 통제 | - 일정 지연<br>- 범위 확대 |
| **종료** | 산출물, 인수인계 | - 최종 보고<br>- 문서화<br>- 교육 | - 지식 유실<br>- 운영 전환 실패 |

---

## 3과목: 데이터 분석 (60점)

### 📊 전체 구조

| 장 | 주제 | 세부 내용 | 중요도 | 문항수(예상) |
|---|------|-----------|--------|-------------|
| 1장 | R/Python 기초 | 기본 문법, 데이터 구조 | ⭐⭐⭐ | 5~8 |
| 2장 | 데이터 전처리 | 정제, 변환, 결측치/이상치 처리 | ⭐⭐⭐⭐⭐ | 12~15 |
| 3장 | 통계 분석 | 기술통계, 추론통계, 가설검정 | ⭐⭐⭐⭐⭐ | 15~18 |
| 4장 | 정형 데이터 마이닝 | 분류, 예측, 군집, 연관분석 | ⭐⭐⭐⭐⭐ | 18~22 |
| 5장 | 비정형 데이터 마이닝 | 텍스트/소셜 분석 | ⭐⭐ | 2~5 |

---

### 1장: R/Python 기초

#### 1.1 Python 데이터 구조

| 자료형 | 특징 | 생성 방법 | 주요 메서드 | 활용 |
|--------|------|-----------|------------|------|
| **List** | - 순서 있음<br>- 변경 가능<br>- 중복 허용 | `[1, 2, 3]` | append(), extend(), insert(), remove() | 일반적인 데이터 저장 |
| **Tuple** | - 순서 있음<br>- 변경 불가<br>- 중복 허용 | `(1, 2, 3)` | count(), index() | 불변 데이터, 함수 반환값 |
| **Set** | - 순서 없음<br>- 변경 가능<br>- 중복 불가 | `{1, 2, 3}` | add(), remove(), union(), intersection() | 중복 제거, 집합 연산 |
| **Dictionary** | - Key-Value<br>- 변경 가능<br>- Key 중복 불가 | `{'a': 1, 'b': 2}` | keys(), values(), items(), get() | 매핑 데이터 |

#### 1.2 NumPy 배열 연산

| 기능 | 메서드/함수 | 설명 | 예시 |
|------|------------|------|------|
| **배열 생성** | `np.array()`, `np.arange()`, `np.zeros()`, `np.ones()` | 다양한 방법으로 배열 생성 | `arr = np.array([1,2,3])` |
| **형태 변경** | `reshape()`, `flatten()`, `transpose()` | 배열 구조 변경 | `arr.reshape(3, 2)` |
| **통계** | `mean()`, `median()`, `std()`, `var()` | 기술 통계량 계산 | `arr.mean()` |
| **집계** | `sum()`, `min()`, `max()`, `argmin()`, `argmax()` | 집계 연산 | `arr.sum(axis=0)` |
| **논리 연산** | `&`, `|`, `~` | 조건 필터링 | `arr[arr > 5]` |

#### 1.3 Pandas 핵심 기능

##### DataFrame 기본 조작

| 작업 | 메서드 | 설명 | 예시 |
|------|--------|------|------|
| **생성** | `pd.DataFrame()` | 딕셔너리, 리스트 등에서 생성 | `df = pd.DataFrame({'A': [1,2,3]})` |
| **조회** | `head()`, `tail()`, `info()`, `describe()` | 데이터 확인 | `df.head(10)` |
| **선택** | `loc[]`, `iloc[]` | 라벨/위치 기반 선택 | `df.loc[0:5, 'A']` |
| **필터링** | 조건식 | 조건에 맞는 행 선택 | `df[df['age'] > 20]` |
| **정렬** | `sort_values()`, `sort_index()` | 값/인덱스 기준 정렬 | `df.sort_values('age')` |

##### 데이터 변환 및 집계

| 기능 | 메서드 | 설명 | 예시 |
|------|--------|------|------|
| **그룹화** | `groupby()` | 그룹별 집계 | `df.groupby('category').mean()` |
| **피벗** | `pivot_table()` | 피벗 테이블 생성 | `pd.pivot_table(df, values='sales', index='month')` |
| **병합** | `merge()`, `concat()`, `join()` | 데이터프레임 결합 | `pd.merge(df1, df2, on='id')` |
| **적용** | `apply()`, `map()`, `applymap()` | 함수 적용 | `df['new'] = df['old'].apply(lambda x: x*2)` |

---

### 2장: 데이터 전처리

#### 2.1 결측치 처리

| 방법 | 함수 | 장점 | 단점 | 적용 상황 |
|------|------|------|------|-----------|
| **삭제** | `dropna()` | - 간단함<br>- 깨끗한 데이터 | - 정보 손실<br>- 샘플 크기 감소 | 결측 비율 낮음 (<5%) |
| **단순 대치** | `fillna(mean/median/mode)` | - 구현 간단<br>- 샘플 크기 유지 | - 분산 왜곡<br>- 관계 무시 | 무작위 결측(MCAR) |
| **예측 대치** | 회귀/KNN 대치 | - 정확도 높음<br>- 관계 보존 | - 계산 복잡<br>- 과적합 위험 | 패턴 있는 결측(MAR) |
| **다중 대치** | `MICE` 알고리즘 | - 불확실성 반영<br>- 통계적 타당성 | - 계산 비용 높음<br>- 해석 복잡 | 중요한 분석 |

#### 2.2 이상치 탐지 및 처리

##### 이상치 탐지 방법

| 방법 | 기준 | 수식 | 특징 | Python 코드 |
|------|------|------|------|-------------|
| **IQR 방법** | 사분위수 기반 | Lower = Q1 - 1.5×IQR<br>Upper = Q3 + 1.5×IQR | - 분포 무관<br>- 보수적 기준 | `Q1 = df['x'].quantile(0.25)`<br>`Q3 = df['x'].quantile(0.75)`<br>`IQR = Q3 - Q1` |
| **Z-score** | 표준편차 기반 | Z = (x - μ) / σ<br>│Z│> 3 | - 정규분포 가정<br>- 민감함 | `from scipy import stats`<br>`z = np.abs(stats.zscore(df['x']))` |
| **Modified Z-score** | MAD 기반 | M = 0.6745(x - median) / MAD<br>│M│> 3.5 | - 로버스트<br>- 이상치에 강함 | `mad = np.median(np.abs(x - np.median(x)))` |
| **Isolation Forest** | 머신러닝 | 고립 정도 측정 | - 고차원 적합<br>- 비선형 탐지 | `from sklearn.ensemble import IsolationForest` |

##### 이상치 처리 방법

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **삭제** | 이상치 제거 | 간단, 명확 | 정보 손실 |
| **대치** | 평균/중앙값 대체 | 샘플 유지 | 분포 왜곡 |
| **변환** | Log, Box-Cox 변환 | 정규화 효과 | 해석 복잡 |
| **캡핑** | 상한/하한값으로 조정 | 정보 보존 | 임의적 |
| **분리 분석** | 이상치 별도 분석 | 패턴 발견 | 추가 작업 |

#### 2.3 데이터 변환

##### 정규화(Normalization) vs 표준화(Standardization)

| 구분 | 정규화 (Min-Max Scaling) | 표준화 (Z-score) |
|------|-------------------------|-----------------|
| **수식** | x' = (x - min) / (max - min) | x' = (x - μ) / σ |
| **범위** | [0, 1] 또는 [-1, 1] | 평균 0, 표준편차 1 |
| **장점** | - 해석 용이<br>- 범위 고정 | - 이상치 영향 적음<br>- 분포 유지 |
| **단점** | - 이상치 민감<br>- 분포 왜곡 | - 범위 불명확 |
| **적용** | 신경망, KNN | 선형회귀, SVM, PCA |
| **Python** | `from sklearn.preprocessing import MinMaxScaler` | `from sklearn.preprocessing import StandardScaler` |

##### 범주형 변수 인코딩

| 방법 | 설명 | 적용 상황 | 코드 예시 |
|------|------|-----------|----------|
| **Label Encoding** | 정수로 매핑 (0, 1, 2, ...) | - 순서형 변수<br>- 트리 기반 모델 | `from sklearn.preprocessing import LabelEncoder`<br>`le.fit_transform(df['city'])` |
| **One-Hot Encoding** | 이진 벡터로 변환 | - 명목형 변수<br>- 선형 모델 | `pd.get_dummies(df, columns=['city'])` |
| **Ordinal Encoding** | 순서 반영 매핑 | 순서가 있는 범주 | `mapping = {'Low':1, 'Medium':2, 'High':3}` |
| **Target Encoding** | 타겟 평균값으로 인코딩 | - 고카디널리티<br>- 트리 모델 | `df.groupby('city')['target'].mean()` |

#### 2.4 피처 엔지니어링

| 기법 | 설명 | 예시 | 효과 |
|------|------|------|------|
| **파생 변수** | 기존 변수 조합으로 새 변수 생성 | 나이 → 연령대, 시간 → 요일/시간대 | 예측력 향상 |
| **다항 특성** | 변수 간 곱셈, 제곱 | x1 × x2, x1² | 비선형 관계 포착 |
| **구간화(Binning)** | 연속형 → 범주형 변환 | 나이 → 10대/20대/30대 | 비선형성, 로버스트 |
| **로그 변환** | 왜도 큰 데이터 변환 | log(매출), log(가격) | 정규분포화 |
| **상호작용 항** | 변수 간 상호작용 효과 | 광고비 × 시즌 | 조건부 효과 |

---

### 3장: 통계 분석

#### 3.1 기술통계

##### 중심 경향성 측도

| 측도 | 정의 | 계산식 | 특징 | 적용 |
|------|------|--------|------|------|
| **평균(Mean)** | 모든 값의 합 / 개수 | μ = Σx / n | - 이상치 민감<br>- 산술 평균 | 대칭 분포 |
| **중앙값(Median)** | 정렬 후 중간값 | 50th percentile | - 이상치에 강건<br>- 위치 통계량 | 왜도 큰 분포 |
| **최빈값(Mode)** | 가장 빈번한 값 | 빈도 최대값 | - 범주형 가능<br>- 다중 모드 가능 | 범주형 데이터 |

##### 산포도 측도

| 측도 | 정의 | 계산식 | 해석 | Python |
|------|------|--------|------|--------|
| **분산(Variance)** | 편차 제곱의 평균 | σ² = Σ(x-μ)² / n | 값의 퍼짐 정도 | `df['x'].var()` |
| **표준편차(SD)** | 분산의 제곱근 | σ = √(σ²) | 원래 단위로 표현 | `df['x'].std()` |
| **범위(Range)** | 최댓값 - 최솟값 | max - min | 이상치 민감 | `df['x'].max() - df['x'].min()` |
| **IQR** | Q3 - Q1 | 75th - 25th percentile | 중간 50% 범위 | `df['x'].quantile(0.75) - df['x'].quantile(0.25)` |
| **변동계수(CV)** | 표준편차 / 평균 | CV = σ / μ | 상대적 변동성 | `df['x'].std() / df['x'].mean()` |

##### 분포 형태 측도

| 측도 | 의미 | 해석 | 계산 |
|------|------|------|------|
| **왜도(Skewness)** | 분포의 비대칭 정도 | - 0: 대칭<br>- 양수: 오른쪽 꼬리<br>- 음수: 왼쪽 꼬리 | `from scipy.stats import skew`<br>`skew(df['x'])` |
| **첨도(Kurtosis)** | 분포의 뾰족한 정도 | - 0: 정규분포<br>- 양수: 뾰족<br>- 음수: 평평 | `from scipy.stats import kurtosis`<br>`kurtosis(df['x'])` |

#### 3.2 확률 분포

##### 이산형 분포

| 분포 | 정의 | 확률질량함수(PMF) | 평균 | 분산 | 활용 |
|------|------|------------------|------|------|------|
| **베르누이<br>(Bernoulli)** | 성공(1) 또는 실패(0) | P(X=1) = p<br>P(X=0) = 1-p | p | p(1-p) | 동전 던지기 |
| **이항<br>(Binomial)** | n번 시행 중 성공 횟수 | P(X=k) = C(n,k) p^k (1-p)^(n-k) | np | np(1-p) | 불량품 개수 |
| **포아송<br>(Poisson)** | 단위 시간당 발생 횟수 | P(X=k) = (λ^k e^(-λ)) / k! | λ | λ | 콜센터 호출 수 |

##### 연속형 분포

| 분포 | 정의 | 확률밀도함수(PDF) | 평균 | 분산 | 활용 |
|------|------|------------------|------|------|------|
| **정규<br>(Normal)** | 종 모양 대칭 분포 | f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²)) | μ | σ² | 키, 시험 점수 |
| **표준정규<br>(Standard Normal)** | μ=0, σ=1인 정규분포 | f(z) = (1/√(2π)) exp(-z²/2) | 0 | 1 | Z-검정 |
| **t 분포** | 소표본 정규분포 | 자유도(df)에 따라 변화 | 0 (df>1) | df/(df-2) | t-검정 |
| **카이제곱<br>(Chi-square)** | 표준정규변수 제곱의 합 | 자유도(k)에 따라 변화 | k | 2k | 적합도 검정 |
| **F 분포** | 두 카이제곱 분산비 | 자유도(d1, d2)에 따라 변화 | d2/(d2-2) | - | 분산 동질성 |

#### 3.3 가설 검정

##### 가설 검정 프로세스

| 단계 | 내용 | 설명 |
|------|------|------|
| **1. 가설 설정** | H₀ (귀무가설)<br>H₁ (대립가설) | - H₀: 차이/효과 없음<br>- H₁: 차이/효과 있음 |
| **2. 유의수준** | α (알파) | 일반적으로 0.05 (5%) |
| **3. 검정통계량** | t, z, χ², F 등 | 데이터로부터 계산 |
| **4. p-value** | 확률값 | H₀ 하에서 관측값이 나올 확률 |
| **5. 의사결정** | 기각 여부 | p < α → H₀ 기각 |

##### 검정 유형 선택 가이드

| 목적 | 데이터 특성 | 검정 방법 | Python 함수 |
|------|------------|----------|-------------|
| **평균 비교 (1표본)** | 정규분포, n≥30 | Z-검정 | `scipy.stats.norm.test()` |
| | 정규분포, n<30 | t-검정 | `scipy.stats.ttest_1samp()` |
| **평균 비교 (2표본)** | 독립, 정규, 등분산 | 독립 t-검정 | `scipy.stats.ttest_ind()` |
| | 독립, 정규, 이분산 | Welch t-검정 | `ttest_ind(equal_var=False)` |
| | 대응(짝지음) | 대응 t-검정 | `scipy.stats.ttest_rel()` |
| **평균 비교 (3표본 이상)** | 독립, 정규, 등분산 | 일원분산분석(ANOVA) | `scipy.stats.f_oneway()` |
| | 비모수 | Kruskal-Wallis | `scipy.stats.kruskal()` |
| **분산 비교** | 두 집단 분산 | F-검정 | `scipy.stats.levene()` |
| | 등분산 검정 | Levene 검정 | `scipy.stats.levene()` |
| **비율 비교** | 독립, 큰 표본 | Z-검정 | `statsmodels.stats.proportion` |
| | 분할표 | 카이제곱 검정 | `scipy.stats.chi2_contingency()` |
| **정규성 검정** | 표본 크기 무관 | Shapiro-Wilk | `scipy.stats.shapiro()` |
| | 큰 표본(n>50) | Kolmogorov-Smirnov | `scipy.stats.kstest()` |
| **상관성 검정** | 정규분포 | Pearson 상관 | `scipy.stats.pearsonr()` |
| | 비모수 | Spearman 상관 | `scipy.stats.spearmanr()` |

##### 오류 유형

| 오류 | 정의 | 확률 | 설명 | 예시 |
|------|------|------|------|------|
| **제1종 오류<br>(Type I Error)** | H₀가 참인데 기각 | α | False Positive<br>(거짓 양성) | 정상인을 환자로 진단 |
| **제2종 오류<br>(Type II Error)** | H₀가 거짓인데 채택 | β | False Negative<br>(거짓 음성) | 환자를 정상으로 진단 |
| **검정력<br>(Power)** | H₀가 거짓일 때 기각 | 1-β | 옳게 기각할 확률 | 환자를 환자로 진단 |

#### 3.4 상관분석과 회귀분석

##### 상관계수

| 유형 | 적용 조건 | 범위 | 해석 | Python |
|------|----------|------|------|--------|
| **Pearson 상관계수** | - 두 변수 정규분포<br>- 선형 관계<br>- 등간/비율 척도 | -1 ~ +1 | │r│> 0.7: 강한 상관<br>0.4~0.7: 중간<br><0.4: 약한 상관 | `df.corr(method='pearson')` |
| **Spearman 상관계수** | - 순서형 데이터<br>- 비선형 단조 관계<br>- 비정규 분포 | -1 ~ +1 | 순위 기반 상관 | `df.corr(method='spearman')` |
| **Kendall's τ** | - 순서형 데이터<br>- 작은 표본 | -1 ~ +1 | 순위 일치도 | `df.corr(method='kendall')` |

##### 단순선형회귀

| 구성 요소 | 수식 | 설명 |
|-----------|------|------|
| **회귀식** | ŷ = β₀ + β₁x | β₀: 절편, β₁: 기울기 |
| **최소제곱법** | Minimize Σ(y - ŷ)² | 잔차 제곱합 최소화 |
| **결정계수(R²)** | R² = 1 - (SSE/SST) | 설명력 (0~1) |
| **조정된 R²** | R²ₐdⱼ = 1 - (1-R²)(n-1)/(n-k-1) | 변수 개수 반영 |

##### 다중선형회귀

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
```

##### 회귀 가정 검증

| 가정 | 검증 방법 | 위반 시 조치 |
|------|----------|-------------|
| **선형성** | 산점도, 잔차도 | 변수 변환, 다항회귀 |
| **독립성** | Durbin-Watson 검정 | 시계열 모델 사용 |
| **등분산성** | Breusch-Pagan 검정 | 가중최소제곱법(WLS) |
| **정규성** | Q-Q plot, Shapiro-Wilk | 변수 변환, 비모수 방법 |
| **다중공선성** | VIF (Variance Inflation Factor) | 변수 제거, PCA |

---

### 4장: 정형 데이터 마이닝

#### 4.1 분류(Classification) 알고리즘

##### 의사결정나무 (Decision Tree)

| 구성 요소 | 설명 | 특징 |
|-----------|------|------|
| **분할 기준** | - Gini 불순도<br>- 정보 이득(Information Gain)<br>- 엔트로피 | - 낮을수록 순수<br>- 높을수록 혼잡 |
| **가지치기** | - 사전 가지치기: max_depth, min_samples_split<br>- 사후 가지치기: cost-complexity pruning | 과적합 방지 |
| **장점** | ✓ 해석 용이<br>✓ 비선형 관계 포착<br>✓ 변수 선택 자동<br>✓ 결측치 처리 | |
| **단점** | ✗ 과적합 경향<br>✗ 불안정(variance 큼)<br>✗ 편향된 분할 | |

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 모델 생성
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# 학습
dt.fit(X_train, y_train)

# 예측 및 평가
y_pred = dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```

##### 앙상블 기법

| 알고리즘 | 방법 | 특징 | 장점 | 단점 | Python |
|----------|------|------|------|------|--------|
| **배깅<br>(Bagging)** | 부트스트랩 샘플링<br>+ 평균/투표 | 병렬 학습<br>분산 감소 | 과적합 감소<br>안정적 | 해석 어려움 | `BaggingClassifier` |
| **랜덤 포레스트<br>(Random Forest)** | 배깅 + 특성 무작위 선택 | 변수 중요도 제공 | 높은 정확도<br>로버스트 | 계산 비용<br>블랙박스 | `RandomForestClassifier` |
| **부스팅<br>(Boosting)** | 순차적 학습<br>오류 집중 | 편향 감소<br>가중치 조정 | 높은 성능 | 과적합 위험<br>느린 학습 | |
| **AdaBoost** | 오분류 가중치 증가 | 약한 학습기 결합 | 간단<br>효과적 | 이상치 민감 | `AdaBoostClassifier` |
| **Gradient Boosting** | 잔차 학습 | 손실함수 최적화 | 매우 높은 성능 | 하이퍼파라미터 많음 | `GradientBoostingClassifier` |
| **XGBoost** | 정규화 + 병렬 처리 | 최적화된 구현 | 빠르고 정확 | 메모리 많이 사용 | `XGBClassifier` |
| **LightGBM** | Leaf-wise 성장 | 대용량 고속 처리 | 매우 빠름 | 과적합 주의 | `LGBMClassifier` |

##### 로지스틱 회귀 (Logistic Regression)

| 구성 요소 | 수식/설명 | 특징 |
|-----------|----------|------|
| **시그모이드 함수** | σ(z) = 1 / (1 + e^(-z)) | 0~1 사이 확률값 |
| **로짓 변환** | logit(p) = log(p/(1-p)) = β₀ + β₁x₁ + ... | 선형 관계로 변환 |
| **최대우도추정** | MLE (Maximum Likelihood Estimation) | 파라미터 추정 |
| **오즈비(Odds Ratio)** | OR = exp(β) | 계수 해석 |

```python
from sklearn.linear_model import LogisticRegression

# 모델 생성
lr = LogisticRegression(
    penalty='l2',  # l1, l2, elasticnet
    C=1.0,  # 정규화 강도 (작을수록 강함)
    solver='lbfgs',
    max_iter=1000
)

# 학습 및 예측
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]  # 양성 확률
```

##### SVM (Support Vector Machine)

| 개념 | 설명 | 특징 |
|------|------|------|
| **서포트 벡터** | 결정 경계에 가장 가까운 데이터 | 모델 결정에 기여 |
| **마진** | 결정 경계와 서포트 벡터 사이 거리 | 최대화가 목표 |
| **커널 트릭** | - Linear<br>- Polynomial<br>- RBF (Gaussian)<br>- Sigmoid | 비선형 분리 가능 |
| **C 파라미터** | 오분류 허용 정도 | 크면 하드 마진<br>작으면 소프트 마진 |

##### 나이브 베이즈 (Naive Bayes)

| 유형 | 가정 | 적용 | 특징 |
|------|------|------|------|
| **Gaussian NB** | 연속형, 정규분포 | 일반 분류 | 빠름, 간단 |
| **Multinomial NB** | 이산형, 빈도 | 텍스트 분류 | 문서 분류 적합 |
| **Bernoulli NB** | 이진형 | 스팸 필터 | 있음/없음 |

#### 4.2 모델 평가

##### 분류 평가 지표

| 지표 | 수식 | 해석 | 사용 시점 |
|------|------|------|----------|
| **정확도<br>(Accuracy)** | (TP + TN) / Total | 전체 중 맞춘 비율 | 균형 데이터 |
| **정밀도<br>(Precision)** | TP / (TP + FP) | 양성 예측 중 실제 양성 | False Positive 비용 큼 |
| **재현율<br>(Recall)** | TP / (TP + FN) | 실제 양성 중 맞춘 비율 | False Negative 비용 큼 |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | 정밀도와 재현율 조화평균 | 불균형 데이터 |
| **특이도<br>(Specificity)** | TN / (TN + FP) | 실제 음성 중 맞춘 비율 | 음성 중요한 경우 |

##### 혼동 행렬 (Confusion Matrix)

|  | 예측: Positive | 예측: Negative |
|--|----------------|----------------|
| **실제: Positive** | TP (True Positive) | FN (False Negative) |
| **실제: Negative** | FP (False Positive) | TN (True Negative) |

##### ROC 곡선 & AUC

| 개념 | 설명 | 해석 |
|------|------|------|
| **ROC 곡선** | TPR(민감도) vs FPR(1-특이도) | 임계값 변화에 따른 성능 |
| **AUC** | ROC 곡선 아래 면적 | 0.5: 무작위<br>0.7~0.8: 괜찮음<br>0.8~0.9: 좋음<br>0.9~: 매우 좋음 |

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ROC 곡선
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

##### 교차 검증 (Cross-Validation)

| 방법 | 설명 | 장점 | 단점 | 사용 |
|------|------|------|------|------|
| **K-Fold CV** | 데이터를 K개로 분할<br>순차적으로 검증 | 모든 데이터 활용<br>안정적 추정 | 계산 비용 | 일반적 상황 |
| **Stratified K-Fold** | 클래스 비율 유지 분할 | 불균형 데이터 대응 | - | 분류 문제 |
| **Leave-One-Out** | n개 중 1개씩 검증 | 최대한 활용 | 계산 비용 매우 큼 | 작은 데이터 |
| **Time Series Split** | 시간 순서 유지 | 시계열 특성 반영 | - | 시계열 데이터 |

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# K-Fold 교차 검증
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

#### 4.3 회귀(Regression) 알고리즘

##### 회귀 평가 지표

| 지표 | 수식 | 범위 | 해석 | 특징 |
|------|------|------|------|------|
| **MAE** | (1/n) Σ│y - ŷ│| [0, ∞) | 평균 절대 오차 | 이상치에 강건 |
| **MSE** | (1/n) Σ(y - ŷ)² | [0, ∞) | 평균 제곱 오차 | 이상치 민감 |
| **RMSE** | √MSE | [0, ∞) | MSE의 제곱근 | 원래 단위로 해석 |
| **R²** | 1 - (SSE/SST) | (-∞, 1] | 설명력 | 1에 가까울수록 좋음 |
| **MAPE** | (1/n) Σ│(y-ŷ)/y│× 100 | [0, ∞) | 평균 절대 백분율 오차 | 상대적 오차 |

##### 정규화 회귀

| 알고리즘 | 정규화 항 | 특징 | 장점 | 사용 |
|----------|----------|------|------|------|
| **Ridge** | L2: α Σβ² | 계수 축소 | 다중공선성 완화 | 모든 변수 유지 |
| **Lasso** | L1: α Σ│β│| 계수를 0으로 | 변수 선택 효과 | 변수 선택 필요 |
| **Elastic Net** | L1 + L2 | 두 가지 결합 | 장점 결합 | 고차원 데이터 |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge 회귀
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso 회귀
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio: L1 비율
elastic.fit(X_train, y_train)
```

#### 4.4 군집(Clustering) 알고리즘

##### K-Means

| 구성 요소 | 설명 | 특징 |
|-----------|------|------|
| **알고리즘** | 1. K개 중심 초기화<br>2. 가까운 중심에 할당<br>3. 중심 재계산<br>4. 수렴까지 반복 | 반복적 최적화 |
| **거리 척도** | 유클리드 거리 | √Σ(x - c)² |
| **K 선택** | - Elbow Method<br>- Silhouette Score<br>- Gap Statistic | 최적 클러스터 수 |
| **장점** | ✓ 간단하고 빠름<br>✓ 대용량 적합 | |
| **단점** | ✗ K 사전 지정<br>✗ 구형 클러스터 가정<br>✗ 이상치 민감 | |

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K 선택 - Elbow Method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 최적 K로 군집화
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# 평가
silhouette = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette:.3f}")
```

##### 계층적 군집(Hierarchical Clustering)

| 방법 | 설명 | 연결 방법 | 특징 |
|------|------|----------|------|
| **응집형<br>(Agglomerative)** | 개별 → 통합 | - Single Linkage (최단)<br>- Complete Linkage (최장)<br>- Average Linkage (평균)<br>- Ward (분산 최소) | 덴드로그램 생성 |
| **분할형<br>(Divisive)** | 전체 → 분할 | Top-down | 계산 복잡 |

##### DBSCAN

| 구성 요소 | 설명 | 특징 |
|-----------|------|------|
| **Epsilon (ε)** | 이웃 반경 | 파라미터 |
| **MinPts** | 최소 점 개수 | 핵심점 기준 |
| **점 유형** | - Core: MinPts 이상<br>- Border: ε 내<br>- Noise: 이상치 | 이상치 탐지 가능 |
| **장점** | ✓ K 불필요<br>✓ 임의 모양<br>✓ 이상치 처리 | |
| **단점** | ✗ 밀도 다르면 어려움<br>✗ 고차원 비효율 | |

##### 군집 평가

| 지표 | 범위 | 해석 | 특징 |
|------|------|------|------|
| **Silhouette Score** | [-1, 1] | 1에 가까울수록 잘 분리 | 내부 응집도 & 외부 분리도 |
| **Davies-Bouldin Index** | [0, ∞) | 낮을수록 좋음 | 클러스터 간 거리/내부 분산 |
| **Calinski-Harabasz** | [0, ∞) | 높을수록 좋음 | 클러스터 간/내 분산 비율 |

#### 4.5 차원 축소

##### PCA (Principal Component Analysis)

| 구성 요소 | 설명 | 특징 |
|-----------|------|------|
| **주성분** | 분산이 최대인 방향 | 고유벡터 |
| **설명 분산** | 각 PC가 설명하는 분산 비율 | 고유값 |
| **차원 선택** | 누적 설명 분산 > 80~90% | 정보 손실 최소화 |
| **장점** | ✓ 다중공선성 제거<br>✓ 시각화<br>✓ 계산 효율 | |
| **단점** | ✗ 해석 어려움<br>✗ 선형 변환만 | |

```python
from sklearn.decomposition import PCA

# PCA 수행
pca = PCA(n_components=0.95)  # 95% 분산 설명
X_pca = pca.fit_transform(X)

# 설명 분산 확인
print(f"Components: {pca.n_components_}")
print(f"Explained Variance: {pca.explained_variance_ratio_}")
```

##### t-SNE

| 특징 | 설명 |
|------|------|
| **목적** | 고차원 → 2D/3D 시각화 |
| **방법** | 확률 분포 유사도 보존 |
| **장점** | 비선형 구조 포착, 시각화 우수 |
| **단점** | 계산 비용 높음, 재현성 낮음 |
| **주의** | 거리 해석 주의, 시각화 전용 |

#### 4.6 연관분석 (Association Rule Mining)

##### 기본 개념

| 용어 | 정의 | 예시 |
|------|------|------|
| **항목집합** | 함께 구매된 상품 집합 | {빵, 우유} |
| **거래(Transaction)** | 한 번의 구매 내역 | 영수증 1장 |
| **규칙** | X → Y | {빵} → {우유} |

##### 평가 지표

| 지표 | 수식 | 의미 | 해석 |
|------|------|------|------|
| **지지도<br>(Support)** | P(X ∩ Y) | X와 Y가 함께 나타나는 비율 | 규칙의 유용성 |
| **신뢰도<br>(Confidence)** | P(Y│X) = P(X ∩ Y) / P(X) | X 구매 시 Y를 구매할 확률 | 규칙의 정확성 |
| **향상도<br>(Lift)** | P(Y│X) / P(Y) | X가 Y 구매에 미치는 영향 | >1: 양의 상관<br>=1: 독립<br><1: 음의 상관 |

##### Apriori 알고리즘

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 데이터 변환
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# 빈발 항목집합 찾기
frequent_items = apriori(df, min_support=0.01, use_colnames=True)

# 연관 규칙 생성
rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

### 5장: 비정형 데이터 마이닝

#### 5.1 텍스트 마이닝

##### 전처리 단계

| 단계 | 작업 | 도구/방법 | 예시 |
|------|------|----------|------|
| **토큰화** | 단어/문장 분리 | KoNLPy, NLTK | "안녕하세요" → ["안녕", "하", "세요"] |
| **불용어 제거** | 의미 없는 단어 제거 | Stopwords | "은", "는", "이", "가" 제거 |
| **어간 추출** | 어간만 추출 | Stemming | "running" → "run" |
| **표제어 추출** | 기본형으로 변환 | Lemmatization | "better" → "good" |
| **정규화** | 대소문자, 특수문자 통일 | lower(), replace() | "HELLO" → "hello" |

##### 벡터화

| 방법 | 설명 | 장점 | 단점 | 코드 |
|------|------|------|------|------|
| **Bag of Words** | 단어 빈도 | 간단, 직관적 | 순서 무시, 희소 | `CountVectorizer` |
| **TF-IDF** | 단어 중요도 | 핵심어 부각 | 문맥 무시 | `TfidfVectorizer` |
| **Word2Vec** | 단어 임베딩 | 의미 유사도 | 학습 필요 | `gensim.Word2Vec` |

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(
    max_features=1000,  # 상위 1000개 단어
    min_df=2,  # 최소 2개 문서 출현
    max_df=0.8,  # 최대 80% 문서 출현
    ngram_range=(1, 2)  # 유니그램 + 바이그램
)

X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
```

##### 감성 분석

| 접근법 | 방법 | 장점 | 단점 |
|--------|------|------|------|
| **사전 기반** | 긍정/부정 단어 사전 | 간단, 해석 용이 | 문맥 무시, 신조어 약함 |
| **머신러닝** | 분류 모델 학습 | 정확도 높음 | 학습 데이터 필요 |
| **딥러닝** | LSTM, BERT | 문맥 이해 | 계산 비용, 데이터 필요 |

#### 5.2 소셜 네트워크 분석

##### 중심성 지표

| 지표 | 의미 | 계산 | 활용 |
|------|------|------|------|
| **연결 중심성** | 직접 연결 수 | 이웃 노드 개수 | 영향력 |
| **근접 중심성** | 전체 거리 합 역수 | 1 / Σ거리 | 정보 전파 |
| **매개 중심성** | 최단 경로 통과 횟수 | 경로 비율 | 브로커 역할 |
| **고유벡터 중심성** | 중요한 노드와 연결 | 가중치 재귀 | 영향력 있는 연결 |

```python
import networkx as nx

# 그래프 생성
G = nx.Graph()
G.add_edges_from([(1,2), (1,3), (2,3), (3,4)])

# 중심성 계산
degree = nx.degree_centrality(G)
closeness = nx.closeness_centrality(G)
betweenness = nx.betweenness_centrality(G)
eigenvector = nx.eigenvector_centrality(G)
```

---

## 학습 로드맵

### 📅 4주 완성 계획

| 주차 | 과목 | 학습 내용 | 목표 | 시간 배분 |
|------|------|----------|------|----------|
| **1주차** | 1과목: 데이터 이해 | - 데이터/정보/지식 개념<br>- 빅데이터 특성<br>- DBMS 유형<br>- SQL 기초 | 기본 개념 완벽 이해 | 주 15시간<br>(평일 2h, 주말 5h) |
| **2주차** | 2과목: 데이터 분석 기획 | - CRISP-DM 방법론<br>- 분석 과제 유형<br>- 거버넌스<br>- 프로젝트 관리 | 방법론 체화 | 주 15시간 |
| **3주차** | 3과목: 데이터 분석 (1) | - Python 기초<br>- 전처리<br>- 통계 분석<br>- 가설 검정 | 통계 + 전처리 마스터 | 주 20시간<br>(평일 3h, 주말 5h) |
| **4주차** | 3과목: 데이터 분석 (2) | - 분류/회귀<br>- 군집/차원축소<br>- 연관분석<br>- 모델 평가 | 머신러닝 알고리즘 숙달 | 주 20시간 |
| **시험 전 1주** | 전 과목 | - 모의고사 3회<br>- 오답 정리<br>- 핵심 요약 복습 | 실전 대비 완성 | 주 25시간 |

### 📚 과목별 학습 전략

#### 1과목: 데이터 이해 (목표 18점/20점)

| 학습 전략 | 세부 내용 |
|----------|----------|
| **이론 숙지** | - 데이터/정보/지식 구분 암기<br>- 빅데이터 3V → 5V 확장 개념<br>- DBMS 유형별 특징 비교표 작성 |
| **실습** | - SQL 기본 쿼리 100문제<br>- JOIN 종류별 실습<br>- 집계 함수 연습 |
| **핵심 암기** | - NoSQL 4가지 유형 특징<br>- ACID vs BASE<br>- 정규화 1~3NF |

#### 2과목: 데이터 분석 기획 (목표 16점/20점)

| 학습 전략 | 세부 내용 |
|----------|----------|
| **방법론 마스터** | - CRISP-DM 6단계 순서 암기<br>- 각 단계별 산출물 정리<br>- KDD, SEMMA와 비교 |
| **과제 유형** | - 최적화/솔루션/인사이트/예측 구분<br>- 각 유형별 적합 기법 매칭 |
| **실무 연계** | - 실제 프로젝트 사례 학습<br>- 우선순위 평가 기준 이해 |

#### 3과목: 데이터 분석 (목표 48점/60점)

| 학습 전략 | 세부 내용 |
|----------|----------|
| **Python 코딩** | - Pandas 데이터 조작 100문제<br>- 전처리 파이프라인 구축<br>- 시각화 기본 (matplotlib, seaborn) |
| **통계 이론** | - 가설검정 플로우차트 작성<br>- 검정 선택 기준표 암기<br>- p-value 해석 연습 |
| **머신러닝** | - 알고리즘별 장단점 비교표<br>- 하이퍼파라미터 튜닝 경험<br>- 평가 지표 선택 기준 |
| **실전 문제** | - 캐글 데이터셋으로 EDA<br>- 모델 비교 프로젝트<br>- 파라미터 영향 분석 |

---

## 실전 준비 전략

### 💡 시험 당일 전략

| 시간 | 활동 | 팁 |
|------|------|-----|
| **0~30분** | 1과목 (20문항) | - 쉬운 문제부터<br>- SQL은 손코딩 연습 필수 |
| **30~60분** | 2과목 (20문항) | - 방법론 순서 문제 주의<br>- 용어 정의 정확히 |
| **60~90분** | 3과목 (50문항) | - 코드 해석 문제 많음<br>- 통계는 계산기 활용<br>- 시간 부족 주의 |

### 🎯 고득점 전략

| 전략 | 세부 사항 |
|------|----------|
| **약점 파악** | 모의고사로 취약 영역 집중 공략 |
| **암기 카드** | 핵심 개념 100개 암기 카드 제작 |
| **오답 노트** | 틀린 문제 유형별 분류 및 재학습 |
| **시간 관리** | 문항당 1분 목표, 어려운 문제는 표시 후 넘김 |
| **실전 연습** | 실제 시험 환경 시뮬레이션 3회 이상 |

### 📖 추천 학습 자료

| 자료 유형 | 추천 |
|----------|------|
| **교재** | - ADsP/ADP 수험서<br>- 파이썬 머신러닝 완벽 가이드 |
| **온라인 강의** | - 데이터 분석 전문가 과정<br>- Coursera Machine Learning |
| **실습** | - Kaggle Learn<br>- LeetCode SQL |
| **모의고사** | - 과년도 기출문제<br>- 모의고사 문제집 |

---

## 핵심 체크리스트

### ✅ 시험 전 최종 점검

- [ ] CRISP-DM 6단계 순서대로 설명 가능
- [ ] SQL JOIN 4가지 유형 차이 명확히 구분
- [ ] Python Pandas 기본 문법 숙지 (loc, iloc, groupby, merge)
- [ ] 가설검정 순서 및 p-value 해석
- [ ] 의사결정나무, 랜덤포레스트, XGBoost 차이
- [ ] 정확도, 정밀도, 재현율, F1-Score 수식
- [ ] K-Means, DBSCAN, 계층적 군집 비교
- [ ] PCA 개념 및 주성분 해석
- [ ] 연관분석 지지도, 신뢰도, 향상도 계산
- [ ] 결측치/이상치 처리 방법 5가지 이상

---

## 마무리

ADP 시험은 **데이터 분석 전 과정**을 다루는 종합 시험입니다:
1. **이론 (1~2과목)**: 개념과 방법론 이해
2. **실전 (3과목)**: Python 코딩 + 통계 + 머신러닝

**핵심은 3과목** - 전체 배점의 60%
- 통계 기초 탄탄히
- Python 실습 많이
- 알고리즘 비교 정리

**꾸준히 학습하고 반복 연습하면 합격 가능합니다!** 🎓

---

**작성일**: 2025년 2월
**버전**: 1.0
**적용 시험**: ADP (데이터분석 전문가) 자격시험
