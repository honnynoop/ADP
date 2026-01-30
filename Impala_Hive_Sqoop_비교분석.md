# Impala, Hive, Sqoop 비교 분석

## 목차
1. [개요](#개요)
2. [Hive (Apache Hive)](#hive-apache-hive)
3. [Impala (Apache Impala)](#impala-apache-impala)
4. [Sqoop (Apache Sqoop)](#sqoop-apache-sqoop)
5. [비교 표](#비교-표)
6. [사용 시나리오](#사용-시나리오)
7. [참고 문헌](#참고-문헌)

---

## 개요

Hadoop 에코시스템에서 Hive, Impala, Sqoop은 서로 다른 목적을 가진 도구들입니다. 각각의 특징을 이해하고 적절한 상황에서 활용하는 것이 중요합니다.

- **Hive**: Hadoop 기반 데이터 웨어하우징 도구 (SQL-like 쿼리 실행)
- **Impala**: 고속 대화형 쿼리 엔진 (실시간 분석)
- **Sqoop**: RDBMS와 Hadoop 간 데이터 전송 도구 (데이터 이관)

---

## Hive (Apache Hive)

### 1.1 정의 및 개발 배경

Apache Hive는 Facebook에서 개발하여 Apache 재단에 기부한 데이터 웨어하우징용 오픈소스 솔루션입니다[^1][^2]. Hadoop 위에서 SQL-like 인터페이스를 제공하여 Java나 Python 코딩 없이도 대용량 데이터를 쿼리할 수 있도록 합니다.

### 1.2 주요 특징

#### 1.2.1 HiveQL (Hive Query Language)
- SQL과 유사한 문법을 사용하여 데이터 분석 가능[^3]
- MapReduce, Apache Tez, Spark 등의 실행 엔진으로 자동 변환됨
- ANSI SQL 표준의 많은 기능 지원 (JOIN, 서브쿼리, 집계 함수, 윈도우 함수 등)

#### 1.2.2 아키텍처 구성요소

```
[Client] → [Driver] → [Compiler] → [Optimizer] → [Executor]
                ↓
         [Metastore]
                ↓
            [HDFS]
```

**핵심 컴포넌트:**

1. **Metastore (메타스토어)**[^4]
   - 테이블 스키마, 파티션, 컬럼 정보 등의 메타데이터 관리
   - 물리적으로는 MySQL, PostgreSQL 등의 RDBMS에 저장
   - 실제 데이터는 HDFS에 저장되어 있음

2. **Driver (드라이버)**
   - HiveQL 쿼리를 받아 전체 프로세스를 제어
   - 컴파일러, 옵티마이저, 실행기를 조율

3. **Compiler (컴파일러)**
   - HiveQL 쿼리를 추상 구문 트리(AST)로 변환
   - AST를 DAG(Directed Acyclic Graph)로 변환
   - MapReduce/Tez/Spark 작업으로 변환

4. **Optimizer (옵티마이저)**[^5]
   - 실행 계획 최적화 (조인 순서, 파티션 프루닝 등)
   - Apache Calcite의 CBO (Cost-Based Optimizer) 사용
   - 쿼리 성능 향상을 위한 다양한 변환 수행

5. **Executor (실행기)**
   - 최적화된 실행 계획을 Hadoop 클러스터에서 실행
   - YARN의 Job Tracker와 상호작용

#### 1.2.3 처리 방식

- **배치 처리 지향**: 대용량 데이터의 ETL, 리포팅, 데이터 분석에 최적화[^6]
- **MapReduce/Tez/Spark 기반**: HiveQL이 이들 프레임워크로 변환되어 실행
- **높은 처리량**: 제한된 리소스에서 효율적으로 대용량 데이터 처리
- **응답 시간**: 수 분에서 수 시간 (대화형 쿼리에는 부적합)

### 1.3 장점

1. **친숙한 SQL 인터페이스**: SQL 경험이 있는 사용자가 쉽게 사용 가능
2. **확장성**: 수평 확장으로 페타바이트급 데이터 처리 가능
3. **다양한 파일 포맷 지원**: Parquet, ORC, Avro, JSON, CSV 등[^7]
4. **파티셔닝 및 버킷팅**: 쿼리 성능 향상
5. **UDF/UDAF/UDTF**: 사용자 정의 함수로 기능 확장 가능
6. **ACID 트랜잭션**: ORC 포맷에서 ACID 지원 (Hive 3.0+)[^8]

### 1.4 단점 및 제약사항

1. **긴 응답 시간**: MapReduce 오버헤드로 인한 지연
2. **실시간 처리 부적합**: OLTP 워크로드에는 사용 불가
3. **데이터 수정 제약**: HDFS 특성상 부분적 Update/Delete 어려움
4. **초기 지연**: 쿼리 실행 전 준비 단계에서 시간 소요

### 1.5 사용 사례

- 대용량 데이터의 배치 처리 및 ETL
- 데이터 웨어하우스 구축
- 로그 분석 및 집계
- 정기적인 리포트 생성
- 데이터 마이닝 및 머신러닝 전처리

---

## Impala (Apache Impala)

### 2.1 정의 및 개발 배경

Apache Impala는 Cloudera에서 개발한 Hadoop 기반의 대화형 SQL 쿼리 엔진입니다[^9]. Google의 Dremel 논문에서 영감을 받아 2012년에 개발되었으며, Hive의 실시간성 문제와 멀티 사용자 지원을 해결하기 위해 만들어졌습니다. 2017년 11월 Apache 재단의 Top-Level 프로젝트가 되었습니다[^10].

### 2.2 주요 특징

#### 2.2.1 고성능 쿼리 엔진

- **MapReduce 미사용**: 자체 분산 쿼리 엔진 사용[^11]
- **C++ 구현**: 인메모리 기반의 고성능 처리
- **MPP (Massively Parallel Processing)**: 대규모 병렬 처리 아키텍처
- **응답 시간**: 수 초 이내 (Hive 대비 10~100배 빠름)[^12]

#### 2.2.2 아키텍처 구성요소

```
[Client] → [Impala Daemon (Coordinator)]
                ↓
    [Impala Daemon (Workers)] × N
                ↓
    [StateStore] + [Catalog Service]
                ↓
         [Hive Metastore]
                ↓
            [HDFS/HBase/Kudu]
```

**핵심 컴포넌트:**[^13][^14]

1. **Impala Daemon (Impalad)**
   - 각 DataNode에서 실행되는 핵심 프로세스
   - 쿼리를 받아 Query Planner, Coordinator, Executor 역할 수행
   - **Data Locality**: 로컬 데이터 직접 읽기로 네트워크 부하 최소화
   - **Direct Read**: HDFS의 데이터를 직접 읽어 처리

2. **StateStore (statestored)**
   - 모든 Impalad의 상태(health) 확인
   - 메타데이터 변경사항을 모든 Impalad에 브로드캐스트
   - 장애 노드 감지 및 전파

3. **Catalog Service (catalogd)**
   - Hive Metastore의 메타데이터 변경사항 관리
   - StateStore를 통해 모든 Impalad에 메타데이터 동기화
   - Impalad에서 직접 변경: 자동 동기화
   - Hive/HDFS에서 변경: `REFRESH` 또는 `INVALIDATE` 명령 필요

#### 2.2.3 Hive와의 호환성

- **HiveQL 호환**: Hive SQL 문법 사용 가능[^15]
- **Hive Metastore 공유**: Hive와 동일한 메타스토어 사용
- **데이터 공유**: Hive에서 생성한 테이블을 Impala에서 직접 쿼리 가능
- **ODBC/JDBC 드라이버**: Hive와 동일한 드라이버 사용 가능

### 2.3 장점

1. **빠른 응답 속도**: 대화형 쿼리 및 Ad-Hoc 분석에 최적화
2. **낮은 레이턴시**: MapReduce 오버헤드 제거
3. **멀티 유저 지원**: 동시 다수 사용자 쿼리 처리
4. **실시간 BI**: 비즈니스 인텔리전스 도구와 통합 용이
5. **메모리 효율성**: 중간 결과를 디스크에 쓰지 않음

### 2.4 단점 및 제약사항

1. **메모리 요구량**: 대용량 쿼리 시 많은 메모리 필요[^16]
2. **장애 허용성**: Hive보다 장애 복구 능력 약함
3. **배치 처리 성능**: 대용량 배치 처리는 Hive/Spark가 더 효율적
4. **데이터 수정 제약**: HDFS 기반이므로 Record별 Update/Delete 불가
5. **설치 복잡도**: Cloudera CDP에 포함, 별도 설치 필요[^17]

### 2.5 성능 벤치마크

TPC-DS 벤치마크 결과 (투이컨설팅, 2021)[^18]:
- Impala 2.8 vs Presto 0.160: Impala 우세
- Impala vs Spark SQL 2.1: Impala 우세
- Impala vs Hive 2.1 with LLAP: Impala 우세
- Single/Multi User 쿼리 응답시간 모두 가장 빠름

### 2.6 사용 사례

- 대화형 데이터 분석 (Ad-Hoc Query)
- 실시간 BI 대시보드
- 탐색적 데이터 분석 (EDA)
- 빠른 프로토타이핑
- Hive로 구조화한 데이터의 대화형 집계

---

## Sqoop (Apache Sqoop)

### 3.1 정의 및 개발 배경

Apache Sqoop은 "SQL-to-Hadoop"의 약자로, RDBMS와 Hadoop 간의 대량 데이터 전송을 위해 설계된 도구입니다[^19]. Cloudera에서 개발하였으며, 2012년 3월 Apache Top-Level 프로젝트가 되었습니다. **2021년 6월 프로젝트가 종료(retired)되어 Apache Attic으로 이동**했습니다[^20].

### 3.2 주요 특징

#### 3.2.1 데이터 전송 방향

```
RDBMS ←→ Hadoop Ecosystem
  ↑          ↓
  ↑    ┌──────────┐
  ↑    │  HDFS    │
  ↑    │  Hive    │
  ↑    │  HBase   │
  ↑    │  Kudu    │
  ↑    └──────────┘
  ↑
Import / Export
```

**Import (RDBMS → Hadoop)**[^21]
- 관계형 데이터베이스의 테이블을 HDFS, Hive, HBase 등으로 임포트
- MapReduce의 Map 태스크를 기반으로 병렬 수행
- 데이터를 파티션으로 나누어 여러 Mapper가 동시 처리

**Export (Hadoop → RDBMS)**
- HDFS의 데이터를 관계형 데이터베이스로 익스포트
- 중간 테이블(staging table) 사용하여 안정성 확보
- 배치 INSERT 실행으로 성능 향상

#### 3.2.2 처리 방식

1. **커넥터 기반**: 각 RDBMS별 커넥터 사용 (JDBC 드라이버 활용)[^22]
2. **병렬 처리**: MapReduce를 통한 병렬 데이터 전송
3. **스키마 자동 인식**: 데이터베이스 스키마를 자동으로 파악
4. **증분 로드 지원**: 변경된 데이터만 선택적 임포트 가능

#### 3.2.3 주요 명령어

**Import 예시:**
```bash
sqoop import \
  --connect jdbc:mysql://host:port/database \
  --username user \
  --password pwd \
  --table table_name \
  --target-dir /user/hadoop/data \
  --split-by id \
  --num-mappers 4
```

**Export 예시:**
```bash
sqoop export \
  --connect jdbc:mysql://host:port/database \
  --username user \
  --password pwd \
  --table table_name \
  --export-dir /user/hadoop/data
```

**Hive 연동 Import:**
```bash
sqoop import \
  --connect jdbc:mysql://host:port/database \
  --table table_name \
  --hive-import \
  --create-hive-table \
  --hive-table db.table_name
```

### 3.3 Sqoop 1 vs Sqoop 2

| 구분 | Sqoop 1 | Sqoop 2 |
|------|---------|---------|
| **아키텍처** | 클라이언트 방식 | 서버-클라이언트 방식 |
| **안정성** | 안정적 (Stable) | 개발 중 (Not production-ready) |
| **최신 버전** | 1.4.7 (2021년 종료) | 1.99.7 (미완성) |
| **호환성** | Sqoop 2와 호환 불가 | Sqoop 1과 호환 불가 |
| **사용 권장** | Production 환경 | 개발/테스트 환경 |

### 3.4 장점

1. **간편한 데이터 이관**: CLI로 간단하게 데이터 전송
2. **병렬 처리**: MapReduce 기반의 고성능 전송
3. **다양한 RDBMS 지원**: Oracle, MySQL, PostgreSQL, SQL Server 등
4. **증분 로드**: 변경 데이터만 선택적 임포트
5. **Hive 통합**: 직접 Hive 테이블로 임포트 가능

### 3.5 단점 및 제약사항

1. **프로젝트 종료**: 2021년 6월 개발 중단 (Apache Attic)[^23]
2. **대안 도구**: Apache NiFi, Apache Flume, Spark JDBC 등 사용 권장
3. **실시간 전송 불가**: 배치 방식의 데이터 전송만 지원
4. **복잡한 변환 제약**: 단순 데이터 이관에 최적화, 복잡한 ETL은 제한적

### 3.6 사용 사례

- 레거시 RDBMS에서 Hadoop으로 데이터 마이그레이션
- 정기적인 데이터 동기화 작업
- 데이터 레이크 구축을 위한 초기 데이터 로드
- Hadoop 처리 결과를 RDBMS로 전송

---

## 비교 표

### 4.1 종합 비교

| 항목 | Hive | Impala | Sqoop |
|------|------|--------|-------|
| **분류** | SQL-on-Hadoop 엔진 | SQL-on-Hadoop 엔진 | 데이터 전송 도구 |
| **개발사** | Facebook → Apache | Cloudera → Apache | Cloudera → Apache |
| **개발 언어** | Java | C++ | Java |
| **주요 목적** | 데이터 웨어하우징, 배치 처리 | 대화형 쿼리, 실시간 분석 | RDBMS ↔ Hadoop 데이터 전송 |
| **실행 엔진** | MapReduce/Tez/Spark | 자체 MPP 쿼리 엔진 | MapReduce |
| **응답 시간** | 수 분 ~ 수 시간 | 수 초 | N/A (전송 도구) |
| **처리 방식** | 배치 처리 | 대화형 처리 | 배치 데이터 전송 |
| **메타스토어** | Hive Metastore (자체) | Hive Metastore (공유) | N/A |
| **데이터 저장소** | HDFS, HBase, S3 등 | HDFS, HBase, Kudu 등 | HDFS ↔ RDBMS |
| **SQL 호환성** | HiveQL (SQL-like) | HiveQL 호환 | N/A (데이터 전송) |
| **트랜잭션** | ACID 지원 (ORC) | 단일 명령문만 지원 | N/A |
| **사용 사례** | ETL, 정기 리포트, 대용량 배치 | Ad-Hoc 쿼리, BI 대시보드 | DB 데이터 이관, 동기화 |
| **프로젝트 상태** | Active (활발히 개발 중) | Active (활발히 개발 중) | Retired (2021년 종료) |
| **최신 버전** | 4.2.0 (2025년) | 4.5.0 (2025년) | 1.4.7 (2016년) |

### 4.2 성능 비교

| 특성 | Hive | Impala | Sqoop |
|------|------|--------|-------|
| **쿼리 응답 속도** | 느림 (분 단위) | 매우 빠름 (초 단위) | N/A |
| **처리량 (Throughput)** | 높음 | 중간 | 높음 (병렬 전송) |
| **레이턴시 (Latency)** | 높음 | 낮음 | N/A |
| **동시 사용자** | 제한적 | 우수 | N/A |
| **대용량 배치** | 최적화됨 | 비효율적 | 최적화됨 |
| **메모리 사용** | 중간 | 높음 | 중간 |
| **디스크 I/O** | 많음 (중간 결과 저장) | 적음 (인메모리) | 많음 (데이터 전송) |

### 4.3 기술적 차이점

| 항목 | Hive | Impala | Sqoop |
|------|------|--------|-------|
| **실행 모델** | MapReduce/Tez/Spark로 변환 | 직접 실행 (No MR) | MapReduce 기반 |
| **중간 결과** | 디스크에 저장 | 메모리에 저장 | 디스크에 저장 |
| **최적화** | CBO (Cost-Based) | CBO + Runtime Optimization | N/A |
| **파일 포맷** | 모든 Hadoop 포맷 | Parquet, ORC, Avro 등 | 텍스트, Avro, Parquet 등 |
| **파티셔닝** | 지원 | 지원 | 지원 (import 시) |
| **압축** | 지원 | 지원 | 지원 |
| **UDF** | 지원 (Java/Python) | 지원 (C++/Java) | N/A |

### 4.4 운영 및 관리

| 항목 | Hive | Impala | Sqoop |
|------|------|--------|-------|
| **설치 난이도** | 쉬움 | 중간 (별도 설치) | 쉬움 |
| **설정 복잡도** | 중간 | 높음 | 낮음 |
| **모니터링** | Web UI, Hue | Web UI, Cloudera Manager | 로그 기반 |
| **장애 허용성** | 높음 (MR 재시도) | 중간 | 중간 (MR 재시도) |
| **리소스 관리** | YARN | YARN 또는 독립 실행 | YARN |
| **보안** | Kerberos, Ranger | Kerberos, Ranger, LDAP | Kerberos |

---

## 사용 시나리오

### 5.1 언제 Hive를 사용하는가?

**적합한 경우:**
- 정기적인 배치 처리 작업 (일/주/월 단위 리포트)
- ETL 파이프라인 구축
- 대용량 데이터의 복잡한 변환 작업
- 비용 효율적인 데이터 웨어하우스 구축
- 장시간 실행되는 쿼리 (수 시간)

**부적합한 경우:**
- 실시간 대화형 쿼리
- 초 단위 응답이 필요한 BI 대시보드
- 트랜잭션 처리 (OLTP)
- 빈번한 데이터 수정/삭제

**예시:**
```sql
-- 월별 매출 집계 (배치 작업)
INSERT OVERWRITE TABLE monthly_sales
SELECT 
    year, month, category,
    SUM(amount) as total_sales,
    COUNT(DISTINCT customer_id) as unique_customers
FROM sales
WHERE year = 2024
GROUP BY year, month, category;
```

### 5.2 언제 Impala를 사용하는가?

**적합한 경우:**
- 대화형 데이터 탐색 (Ad-Hoc Query)
- 실시간 BI 대시보드 (Tableau, Power BI 연동)
- 빠른 프로토타이핑 및 분석
- 멀티 유저 환경의 동시 쿼리
- Hive로 만든 데이터의 빠른 조회

**부적합한 경우:**
- 매우 복잡한 대용량 배치 처리
- 메모리가 부족한 환경
- 높은 안정성이 요구되는 미션 크리티컬 배치

**예시:**
```sql
-- 실시간 고객 행동 분석
SELECT 
    product_category,
    AVG(session_duration) as avg_duration,
    COUNT(*) as total_sessions
FROM user_sessions
WHERE date = CURRENT_DATE()
GROUP BY product_category
ORDER BY total_sessions DESC
LIMIT 10;
```

### 5.3 언제 Sqoop을 사용하는가?

**적합한 경우:**
- 레거시 RDBMS에서 데이터 레이크로 마이그레이션
- 정기적인 데이터 동기화 (일 단위 증분 로드)
- 초기 데이터 로드 (One-time migration)
- Hadoop 분석 결과를 RDBMS로 전송

**부적합한 경우:**
- 실시간 데이터 스트리밍 (→ Kafka, Flume 사용)
- 복잡한 데이터 변환 (→ Spark, Hive 사용)
- 2021년 이후 신규 프로젝트 (프로젝트 종료됨)

**대안 도구:**
- **Apache NiFi**: 데이터 흐름 자동화
- **Apache Flume**: 실시간 로그 수집
- **Spark JDBC**: Spark를 이용한 RDBMS 연동
- **Talend, Informatica**: 상용 ETL 도구

**예시:**
```bash
# MySQL에서 Hive로 일별 증분 데이터 임포트
sqoop import \
  --connect jdbc:mysql://db.example.com:3306/sales \
  --username sqoop_user \
  --password-file /user/sqoop/password.txt \
  --table orders \
  --hive-import \
  --hive-table sales.orders \
  --incremental lastmodified \
  --check-column last_update \
  --last-value "2024-01-30 00:00:00" \
  --split-by order_id \
  --num-mappers 8
```

### 5.4 통합 사용 시나리오

**일반적인 빅데이터 파이프라인:**

```
[RDBMS] 
   ↓ (Sqoop Import - 일 1회)
[HDFS Raw Data]
   ↓ (Hive ETL - 배치 처리)
[HDFS Processed Data / Hive Tables]
   ↓ (Impala - 실시간 쿼리)
[BI Dashboard / Analytics]
   ↓ (Sqoop Export - 필요 시)
[RDBMS Summary Tables]
```

**단계별 설명:**

1. **데이터 수집**: Sqoop으로 RDBMS → HDFS
2. **데이터 가공**: Hive로 복잡한 ETL 및 데이터 정제
3. **실시간 분석**: Impala로 대화형 쿼리 및 BI 연동
4. **결과 저장**: 필요 시 Sqoop으로 HDFS → RDBMS

**예시 시나리오: 이커머스 분석 플랫폼**

```bash
# 1. Sqoop: 매일 00:00에 주문 데이터 수집
sqoop import --connect jdbc:mysql://... --table orders --incremental append

# 2. Hive: 데이터 정제 및 집계 (배치)
# orders_raw → orders_cleaned → daily_summary
hive -e "
INSERT INTO TABLE daily_summary
SELECT date, category, SUM(amount), COUNT(*)
FROM orders_cleaned
WHERE date = '2024-01-30'
GROUP BY date, category;
"

# 3. Impala: 실시간 대시보드 쿼리 (초 단위 응답)
impala-shell -q "
SELECT category, SUM(total_amount) 
FROM daily_summary 
WHERE date BETWEEN '2024-01-01' AND '2024-01-30'
GROUP BY category;
"
```

---

## 참고 문헌

### 공식 문서

#### Apache Hive
[^1]: Apache Hive 공식 홈페이지, https://hive.apache.org/
[^2]: Apache Hive Documentation, https://cwiki.apache.org/confluence/display/Hive/
[^3]: Apache Hive GitHub Repository, https://github.com/apache/hive
[^4]: Apache Hive - Wikipedia, https://en.wikipedia.org/wiki/Apache_Hive
[^5]: Apache Hive - Cost-based optimization, https://hive.apache.org/docs/latest/
[^6]: Apache Hive - GeeksforGeeks, https://www.geeksforgeeks.org/devops/apache-hive/
[^7]: Apache Hive User Manual, https://hive.apache.org/docs/latest/user/
[^8]: Hive 4.0 - Overview of Major Changes, https://hive.apache.org/docs/latest/

#### Apache Impala
[^9]: Apache Impala 공식 홈페이지, https://impala.apache.org/
[^10]: Apache Impala Documentation, https://impala.apache.org/impala-docs.html
[^11]: Apache Impala Overview, https://impala.apache.org/overview.html
[^12]: Cloudera Impala Documentation, https://docs.cloudera.com/runtime/7.2.18/impala-overview/
[^13]: Introducing Apache Impala, https://impala.apache.org/docs/build/html/topics/impala_intro.html
[^14]: Apache Impala GitHub Repository, https://github.com/apache/impala
[^15]: Apache Impala SQL Reference, https://docs.cloudera.com/cdw-runtime/cloud/impala-sql-reference/
[^16]: eG Innovations - What is Apache Impala?, https://www.eginnovations.com/documentation/Apache-Impala/

#### Apache Sqoop
[^19]: Apache Sqoop 공식 홈페이지, https://sqoop.apache.org/
[^20]: Sqoop - Wikipedia, https://en.wikipedia.org/wiki/Sqoop
[^21]: Sqoop User Guide (v1.4.6), https://sqoop.apache.org/docs/1.4.6/SqoopUserGuide.html
[^22]: Apache Sqoop Documentation, https://sqoop.apache.org/docs/1.99.7/index.html
[^23]: Apache Attic - Sqoop (프로젝트 종료), 2021년 6월

### 기술 블로그 및 분석 자료

[^17]: 투이컨설팅 (2021), "SQL On Hadoop 분석 도구인 Hive와 Impala는 어떤 차이가 있을까?", https://www.2e.co.kr/news/articleView.html?idxno=301587
[^18]: TPC-DS 벤치마크 - Impala vs Presto vs Spark SQL vs Hive, 투이컨설팅, 2021

### 한국어 기술 문서
- 빅데이터 - 하둡, 하이브로 시작하기, https://wikidocs.net/22651
- [Hadoop] Hadoop Ecosystem과 구성요소들, https://yganalyst.github.io/hadoop/hdp_ecosys/
- Hadoop, Hive, Spark에 대해 자세히 알아보기, https://seoyoungh.github.io/data-science/distribute-system-1/
- 네이버 클라우드 - Sqoop 사용 가이드, https://docs.ncloud.com/ko/hadoop/chadoop-4-1.html
- 네이버 클라우드 - Impala 사용, https://guide.ncloud-docs.com/docs/hadoop-vpc-30

---

## 요약

### 핵심 차이점 한 줄 정리

- **Hive**: "대용량 배치 처리의 왕" - 느리지만 안정적이고 효율적
- **Impala**: "실시간 분석의 강자" - 빠르지만 메모리가 필요함
- **Sqoop**: "데이터 이관의 도구" - RDBMS ↔ Hadoop 브릿지 (현재 종료됨)

### 선택 가이드

```
질문 1: 무엇을 하고 싶은가?
├─ 데이터 전송 (RDBMS ↔ Hadoop)
│  └─ Sqoop (또는 대안: NiFi, Flume)
│
└─ 데이터 쿼리/분석
   │
   ├─ 질문 2: 응답 시간이 얼마나 중요한가?
   │
   ├─ 수 초 이내 (실시간 대화형)
   │  └─ Impala 사용
   │
   └─ 수 분~수 시간 (배치 처리)
      └─ Hive 사용
```

### 빅데이터분석기사 시험 대비 핵심 키워드

**Hive:**
- MapReduce/Tez/Spark 기반
- HiveQL (SQL-like)
- Metastore (스키마 관리)
- 배치 처리, ETL
- ACID 트랜잭션 (ORC)

**Impala:**
- MPP (Massively Parallel Processing)
- C++ 인메모리 엔진
- MapReduce 미사용
- 대화형 쿼리, Ad-Hoc 분석
- Hive Metastore 공유

**Sqoop:**
- SQL-to-Hadoop
- RDBMS ↔ Hadoop 데이터 전송
- MapReduce 기반 병렬 처리
- Import/Export
- 2021년 프로젝트 종료

---

**문서 작성일**: 2025-01-30  
**작성자**: Claude (Anthropic)  
**목적**: 빅데이터분석기사 자격증 준비용 학습 자료
