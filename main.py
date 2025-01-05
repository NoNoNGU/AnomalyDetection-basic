import numpy as np
import pandas as pd
import joblib
import argparse
import time
from datetime import datetime
from elasticsearch import Elasticsearch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

def connect_ea():
    es = Elasticsearch('http://localhost:9200')
    if es.ping():
        print('[*] Successfully connected to Elasticsearch')
        return es
    else:
        print('[!] Failed to connect to Elasticsearch')
        return None


def fetch_logs_for_train(es, timestamp):
    logs = []
    docs = es.search(index='nginx-logs-*', body={
        "query": {
            "range": {
                "@timestamp": {"lt": timestamp}
            }
        },
        "sort": {
            "@timestamp": {"order": "desc"}
        }
    }, scroll='1m', size=100)
    scroll_id = docs['_scroll_id']
    n_docs = len(docs['hits']['hits'])
    logs += [d['_source'] for d in docs['hits']['hits']]
    cnt = n_docs
    while n_docs > 0:
        docs = es.scroll(scroll_id=scroll_id, scroll='1m')
        logs += [d['_source'] for d in docs['hits']['hits']]
        n_docs = len(docs['hits']['hits'])
        cnt += n_docs

    print(f'{cnt} docs retrieved')
    es.clear_scroll(scroll_id=scroll_id)
    return logs


def fetch_logs_for_detection(es, start_time, end_time):
    logs = []
    docs = es.search(index='nginx-logs-*', body={
        "query": {
            "range": {
                "@timestamp": {
                    "gt": start_time,
                    "lte": end_time
                }
            }
        },
        "sort": {
            "@timestamp": {"order": "desc"}
        }
    }, scroll='1m', size=100)
    scroll_id = docs['_scroll_id']
    n_docs = len(docs['hits']['hits'])
    logs += [d['_source'] for d in docs['hits']['hits']]
    cnt = n_docs
    while n_docs > 0:
        docs = es.scroll(scroll_id=scroll_id, scroll='1m')
        logs += [d['_source'] for d in docs['hits']['hits']]
        n_docs = len(docs['hits']['hits'])
        cnt += n_docs
    print(f'{cnt} docs retrieved')
    es.clear_scroll(scroll_id=scroll_id)
    return logs


def transform_to_dataset(logs, do_save=False):
    data_rec = []
    data = []
    for log in logs:
        if '_grokparsefailure' not in log['tags']:
            timestamp = log['@timestamp']
            method = log['method']
            uri = log['request']
            status_code = log['response_code']
            user_agent = log['user_agent']
            client_ip = log['client_ip'].split('.')
            client_ip0 = int(client_ip[0])
            client_ip1 = int(client_ip[1])
            client_ip2 = int(client_ip[2])
            client_ip3 = int(client_ip[3])
            data_rec.append([method, uri, status_code, user_agent, client_ip0, client_ip1, client_ip2, client_ip3])
            data.append([timestamp, method, uri, status_code, user_agent, log['client_ip']])

    columns_rec = ['method', 'uri', 'status_code', 'user_agent', 'client_ip0', 'client_ip1', 'client_ip2', 'client_ip3']
    columns = ['@timestamp', 'method', 'uri', 'status_code', 'user_agent', 'client_ip']

    data_rec = pd.DataFrame(data_rec, columns=columns_rec)
    data = pd.DataFrame(data, columns=columns)

    if do_save:
        print('[*] Successfully saved dataset')
        data_rec.to_csv('./data.csv', index=False)

    return data_rec, data


def preprocess(data):
    print('[*] Preprocessing data')

    # 범주형 데이터 인코딩
    categorical_cols = ['method', 'user_agent']
    for col in categorical_cols:
        if col in data.columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    # 정규화
    for col in ['client_ip0', 'client_ip1', 'client_ip2', 'client_ip3']:
        if col in data.columns:
            data = normalize(data, col, 0, 255)

    # 표준화
    if 'status_code' in data.columns:
        data = standardize(data, 'status_code')

    # 숫자 데이터만 선택
    numeric_cols = data.select_dtypes(include=['number']).columns
    data = data[numeric_cols]

    return data

def normalize(data, col, min_val, max_val):
    data[[col]] = (data[[col]] - min_val) / (max_val - min_val)
    return data


def standardize(data, col):
    scaler = StandardScaler()
    data[[col]] = scaler.fit_transform(data[[col]])
    return data


def train_detector(data):
    clf = IsolationForest(contamination=0.01)
    print('[*] Fit anomaly detector')
    clf.fit(data)

    res = pd.DataFrame(clf.predict(data))  # 1: normal, -1: anomaly
    print(f'the number of anomaly: {res.value_counts().get(-1, 0)}')
    print(f'the number of normal: {res.value_counts().get(1, 0)}')

    print('[*] Save the trained model')
    joblib.dump(clf, 'detector.pkl')


def load_detector():
    clf = joblib.load('detector.pkl')
    return clf


def detect_anomaly(es):
    clf = load_detector()
    with open('lasttime', 'r') as f:
        start_time = f.readline().strip('')
    while True:
        now = datetime.utcnow()
        cur_time = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        logs = fetch_logs_for_detection(es, start_time, cur_time)
        if not logs:
            time.sleep(1)
            continue

        with open('lasttime', 'w') as f:
            f.write(cur_time)

        data_rec, data = transform_to_dataset(logs)
        data_p = preprocess(data_rec)

        res = pd.DataFrame(clf.predict(data_p), columns=['pred'])['pred'].map({-1: 'anomaly', 1: 'normal'})

        data = pd.concat([data, res], axis=1)

        data = data.to_dict(orient='records')
        for record in data:
            es.index(index='anomaly-nginx', body=record)

        start_time = cur_time
        time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default='detect', choices=['train', 'detect'])
    return parser.parse_args()


def main():
    args = parse_args()
    es = connect_ea()

    if args.job == 'train' :
        now = datetime.utcnow()
        timestamp = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        with open('lasttime', 'w') as f:
         f.write(timestamp)

        logs = fetch_logs_for_train(es, timestamp)
        transform_to_dataset(logs, True)

        data = pd.read_csv('./data.csv')
        data = preprocess(data)
        train_detector(data)
    elif args.job == 'detect' :
        detect_anomaly(es)


if __name__ == "__main__":
    main()