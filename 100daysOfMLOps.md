### Task 20

```bash
### run the mlflow server
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*" \
    --backend-store-uri sqlite:////root/code/mlflow-backend/mlflow.db \
    --default-artifact-root /root/code/mlflow-artifacts/
```


### Task 21

```bash
#TODO_1
mlflow.log_params(params) # stores dictonary or list

#TODO_2
mlflow.log_metric("accuracy", accuracy)  ## to log the metrics 
mlflow.log_metric("f1_score", f1)

#TODO_3
mlflow.sklearn.log_model(sk_model=model) # Stores model information & that version artifacts

```


### Task 22

- Using of UI , create 2 experiment , add tags 


### Task 23

- Comapre the experiments & add tags by searching through the query in the UI


### Task 24

- `mlflow.sklearn.autolog()` // Set autolog for a specific package
- `mlflow.set_experiment("autolog-demo")` // Set the experiment


### Task 25
Register , version & manage model lifecycle

- Create a model registry
- Register your model from the UI
- Give alias
- Compare the model


### Task 26
Compare model runs & select the best

```python
# TODO1
signature = mlflow.models.infer_signature(X, preds)

# TODO2
mlflow.sklearn.log_model(
    model,
    name="model",
    input_example=X,      # Generates required signature
    signature=signature,  # Or provide explicit signature
    )
```

### Task 27
Load model from the registry with custom pre-processing

```python
# TODO1
        X_scaled = (X - self.mean) / self.std
        return self.model.predict(X_scaled)

# TODO2
inner_model = mlflow.pyfunc.load_model(MODEL_URI)

# TODO3
preds = predictor.predict(None, inputs.values) 
inputs["prediction"] = np.asarray(preds).reshape(-1)
inputs.to_csv(OUTPUT_CSV, index=False)

```

### Day 28
Debug mlflow project & re-run it

``` shell
root@controlplane ~/code/trainer via 🐍 v3.12.3 ➜  cat /tmp/mlflow-run-initial.log
2026/07/12 07:11:40 INFO mlflow.projects: 'trainer' does not exist. Creating a new experiment
2026/07/12 07:11:40 INFO mlflow.projects.utils: === Created directory /tmp/tmpkgj8wag3 for downloading remote URIs passed to arguments of type 'path' ===
2026/07/12 07:11:40 INFO mlflow.projects.backend.local: === Running command 'python train.py --n_est 100' in run with ID '9a46ca1bca7841b98c4c047cedf71201' === 
usage: train.py [-h] [--n_estimators N_ESTIMATORS] [--max_depth MAX_DEPTH]
                [--test_size TEST_SIZE] [--random_seed RANDOM_SEED]
train.py: error: unrecognized arguments: --n_est 100
2026/07/12 07:11:42 ERROR mlflow.cli: === Run (ID '9a46ca1bca7841b98c4c047cedf71201') failed ===
🏃 View run rebellious-snake-745 at: http://localhost:5000/#/experiments/1/runs/9a46ca1bca7841b98c4c047cedf71201
🧪 View experiment at: http://localhost:5000/#/experiments/1

```

`Edit the MLproject`
```yaml
command: >
      python train.py
      --n_estimators {n_estimators}
      --max_depth {max_depth}
      --test_size {test_size}
      --random_seed {random_seed}
```

```shell
# Run the foloowing command
mlflow run . -e train --env-manager local
mlflow run . -e train -P n_estimators=200 -P max_depth=10 --env-manager local
```



### Day 29
Fix MLflow's Remote Artifact-Store Wiring (PostgreSQL + SeaweedFS)


```shell
# Debug commands for the postgresQL
docker exec mlflow-db psql -U mlflow -d mlflow --version
docker exec -it mlflow-db psql -U mlflow -d mlflow 
```


```shell
# Added/Edited these lines in startup script
export MLFLOW_S3_ENDPOINT_URL="http://localhost:8333"

# Added --serve-artifacts 
exec mlflow server \
  --backend-store-uri postgresql://mlflow:mlflow123@localhost:5432/mlflow \
  --artifacts-destination s3://mlflow-artifacts \
  --serve-artifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts '*' --cors-allowed-origins '*'

```


### Day 30
End-to-End MLflow: Register, Serve, and Monitor the Champion



Register the model from the UI , with given format & criteria

```shell
export MLFLOW_TRACKING_URI=http://localhost:5000

mlflow models serve -m models:/fraud-detector-v2/1 -p 5001 --env-manager=local
```


```curl localhost:5001/health -vvv```


**monitor.sh**

```shell
#!/usr/bin/env bash
set -u

URL="http://127.0.0.1:5001/health"

code=$(curl -s -o /dev/null -w '%{http_code}' "$URL")
if [ "$code" = "200" ]; then
  exit 0
else
  exit 1
fi
```



### Day 31
Fix a Broken Config-Driven Training Setup

Edit in **train_config.yaml**
```yaml
type: RandomForestClassifier
```
```yaml
data:
  target_column: is_fraud
```
```yaml
output:
  model_path: /root/code/fraud-detection/models/model.pkl
```


### Day 32
Make a Training Script Reproducible (Seed Discipline)


```python
# Added random_state in split & model calling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```

### Day 33
Fix a Broken Evaluation Script and Metrics Report

update the `train.py`

```python
METRICS_JSON = os.path.join(REPORTS_DIR, "metrics.json")
```
```python
    metrics = {
        "accuracy": round(accuracy_score(y, preds), 6),
        "precision": round(precision_score(y, preds), 6),
        "recall": round(recall_score(y, preds), 6),
        "f1_score": round(f1_score(y, preds), 6),
        "auc_roc": round(roc_auc_score(y, preds), 6)
    }
```




### Day 34
Fix a Broken Cross-Validation Loop (Stratified + Aggregates)
```python
# TODO 1
fold = {
    "fold": fold_idx,
    "accuracy": round(accuracy_score(y_test, preds), 6),
    "f1": round(f1_score(y_test, preds), 6),
    "roc_auc": round(roc_auc_score(y_test, proba), 6),
}

# TODO 2
with mlflow.start_run(run_name=f"fold-{fold_idx}", nested=True):
    mlflow.log_param("fold", fold_idx)
    mlflow.log_metric("accuracy", fold["accuracy"])
    mlflow.log_metric("f1", fold["f1"])
    mlflow.log_metric("roc_auc", fold["roc_auc"])
```



### Day 35 (Important)
Fix a Broken Optuna Tuner with MLflow Logging

```python
# Fix 1 : Track the log for each trial
with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("f1_score", score)
```

```python
# Fix 2: Replace the minimize to maxmimze , as we need the value for the highest
study = optuna.create_study(direction="maximize", study_name=EXPERIMENT_NAME)
```




### Day 36
Fix a Multi-Model Bake-Off in the MLflow Compare View


```python
    # Replace ASC to DESC as we need the highest f1_score
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=10,
    )
```
```python
    winner = runs.iloc[0]
    report = {
        "run_id": winner["run_id"],
        "f1_score": float(winner["metrics.f1_score"]),
        # Added the model_type to the json
        "model_type": winner["tags.mlflow.runName"]
    }
```



### Day 37 (Important)
Fix a Four-Stage Training Pipeline's Inter-Stage Wiring

```python
# featurize.py
input_path = config["data"]["processed_path"]
```
```python
train.py
features_path = config["data"]["features_path"]
```

```python
# run_pipeline.py
# TODO 1
mlflow.log_param("model_type",config["model"]["type"])
mlflow.log_param("n_estimators",config["model"]["n_estimators"])
mlflow.log_param("max_depth",config["model"]["max_depth"])


# TODO 2
metrics = config["output"]["report_path"]
with open(metrics, "r") as file:
    data = json.load(file)
mlflow.log_metrics(data)

```



### Day 38:
Fix a Parallel-Training Bake-Off (n_jobs Backend)


```python
# train.py
N_JOBS_VALUES = [1, -1]
mlflow.log_param("n_jobs", n_jobs)


with mlflow.start_run(run_name="speedup-summary"):
        mlflow.log_metric("speedup",(times[1] / times[-1]))
```


### Day 39: (Good one) - you can run on CPU or in GPU based upon your needs
Make a PyTorch Trainer Device-Aware with Checkpointing

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = model.to(device)


mlflow.log_param("device", device)
xb = X_t.to(device)
yb = y_t.to(device)

```
```python
# TODO
if epoch % 10 == 0:
    torch.save(
    {'epoach': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': final_loss
    },
    os.path.join(CHECKPOINT_DIR,f"ckpt_epoch_{epoch}.pt")
    )

```


### Day 40: 
Fix and Complete a Five-Stage Training Capstone



```shell
  # Makefile
  # Re-arrange the file 
  python3 src/validate_data.py
  python3 src/tune.py
  python3 src/select_model.py
  python3 src/register.py
  python3 src/report.py
```

```python
# select_model.py
# Replaced accuracy to f1_score 
runs = mlflow.search_runs(
      experiment_ids=[exp.experiment_id],
      order_by=["metrics.f1_score DESC"],
      max_results=200,
  )

score = float(best["metrics.f1_score"])

```

```python
# register.py
# TODO
client.set_registered_model_alias(REGISTERED_MODEL_NAME, RELEASE_ALIAS, version.version)
```

```python
# report.py
report = {
    "best_model": selection["model_type"],
    "best_params": best_params,
    "metrics": best_metrics,
    "total_trials": total_trials,
    "validation_status": validation["status"]
}
```


### Day 41
Scaffold a Feast Feature Repository and Build a Training Set

```shell
feast init feature_repo
cd feature_repo/feature_repo
feast apply
```

```shell
feast ui & // to run the Feast UI
```


```python
# replace training_df line with this 
training_df = store.get_historical_features(entity_df=entity_df, features=["driver_hourly_stats:conv_rate", "driver_hourly_stats:acc_rate", "driver_hourly_stats:avg_daily_trips"]).to_df()

```

``` shell
python  build_training_set.py 
```

### Day 42
Define a Feast Feature View (Entity + Field Schema)

```python
join_keys=["customer_id"],
```
```python 
Field(name="hour", dtype=Int64),
Field(name="num_tx_past_day", dtype=Int64),
```




### Day 43


```shell
# ./materialize.sh
END_DATE="2026-01-01T00:00:00"
```


```python
# fetch_features.py
result = store.get_online_features(
    features=[
      "customer_transaction_features:amount",
      "customer_transaction_features:hour",
      "customer_transaction_features:num_tx_past_day",
  ] , entity_rows=[{"customer_id": i} for i in range(1, 6)]   
).to_dict()
```

### Day 44 
> Store MLflow's Admin Password in HashiCorp Vault

- Loging to the vault UI (usbing the vault token given to us)
- Create the Secret Engine (with version: 2)
- Create a secret in `mlflow` path & create `admin_password` secret with random password


### Day 45 (Hard)
> Authenticate MLflow to Vault via AppRole and Fix Its KV Policy

- Edit the ACL policy
```shell
path "secret/data/mlflow" {
  capabilities = ["read"]
}
```


```shell
export VAULT_ADDR=http://127.0.0.1:8200
export VAULT_TOKEN=$(cat /root/code/vault-root-token)

vault auth enable approle

vault write auth/approle/role/mlflow \
  token_policies=mlflow-reader \
  token_ttl=1h token_max_ttl=4h

```



### Day 46
> Author Data-Quality Expectations with Great Expectations


```python
suite.add_expectation(ge.ExpectTableColumnsToMatchSet (column_set=["amount","hour","num_tx_past_day","is_fraud"]))
```

```python
suite.add_expectation(ge.ExpectColumnValuesToBeBetween(column="amount" , min_value=0))
```

```python
suite.add_expectation(ge.ExpectColumnValuesToBeBetween(column="hour",min_value=0,max_value=23))
```

```python
suite.add_expectation(ge.ExpectColumnValuesToBeInSet(column="is_fraud", value_set=[0,1]))
```





### Day 47
> Debug a Failing Great Expectations Checkpoint

```python
# replace the 0 with -400 based on the result from the run in UI
ge.ExpectColumnValuesToBeBetween(column="amount", min_value=-400)
```


### Day 48
> Enforce a Data-Quality Checkpoint as a Blocking CI Gate


Go to the workflow in `.gitea/workflows/data-quality` , & add the below step

```yaml
- name: running the command 
  run: python3 -m src.gx_run
```

```shell
# commit & push
git add .
git commit -m "Added CI"
git push
```

### Day 49 (GOOD one )
> Secrets + Data-Quality Integration Capstone


Step 1 : create a mlflow_password inside vault UI


Step 2 : 
```shell
TOKEN=$(cat /root/code/vault-token)
PASSWORD=$(curl -sf -H "X-Vault-Token: $TOKEN" \
  "$VAULT_ADDR/v1/secret/data/mlflow" \
  | python3 -c "import json, sys; print(json.load(sys.stdin)['data']['data']['mlflow_password'])")
if [ -z "$PASSWORD" ]; then
  echo "::error::Empty password from Vault -- stage mlflow_password in secret/mlflow first"
  exit 1
fi
echo "::notice::Fetched MLflow password from Vault (len=${#PASSWORD})"

```

```shell
# commit & push
git add .
git commit -m "Added CI"
git push
```

- Create a PR & merge the PR
- Check the actions , It should be successfully ran
- login to MLFlow UI , go to model and add Alias as `Production`



### Day 50
> Create Docker Image for ML Training Environment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install scikit-learn pandas numpy joblib
COPY train.py /app/train.py
CMD ["python3" , "train.py"]
```
```shell
# build the image
docker build -t ml-trainer:v1 .
```
```shell
# to validate
docker run --rm ml-trainer:v1 python3 -c "import sklearn, pandas, numpy, joblib; print('OK')" prints OK
```



### Day 51
> Create Multi-Stage Docker Build for ML Serving

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir scikit-learn pandas numpy joblib flask
COPY train_model.py /app/train_model.py
RUN python3 /app/train_model.py



FROM  python:3.11-slim AS runtime
WORKDIR /app
RUN pip install --no-cache-dir scikit-learn numpy joblib flask
COPY --from=builder /app/model.pkl /app/model.pkl
COPY serve.py /app/serve.py
EXPOSE 8080
CMD ["python3", "/app/serve.py"]

```

```shell
# build the image
docker build -t ml-serve:v1 .
```


```shell
# Validate it 
docker images ml-serve:v1 lists the built image; docker run --rm -p 8090:8080 ml-serve:v1

# On new tab
curl localhost:8090/health
```



### Day 52
> Fix a Broken Jupyter + MLflow + SeaweedFS Compose Stack

```yaml
jupyter
  command: "start-notebook.sh --ServerApp.token='' --ServerApp.password=''"
seaweedfs
      - "9000:8333" # S3 API
      - "9001:8888" # Filer UI
```

```shell
docker compose up -d 
docker compose ps
```

```
verify by opening all the 3 UI
```



### Day 53
> Fix a Broken PyTorch Dockerfile (CPU-Wheel URL)


```dockerfile
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

CMD ["python3", "-c", "import torch; print(torch.__version__, 'cuda?', torch.cuda.is_available())"]

```


```shell
# build the image
docker build -t dl-trainer:v1 .
```

```shell
# list the image
docker images dl-trainer:v1
```

```shell
# verify the image
docker run --rm dl-trainer:v1
```



### Day 54
> Push ML Model Images to Container Registry

```shell
REGISTRY="localhost:5555"
docker tag "$IMAGE" "$REGISTRY/$IMAGE"
docker push "$REGISTRY/$IMAGE"
```

```shell
# validate 
root@controlplane ~/code/ml-registry via 🐍 v3.12.3 ➜ curl http://localhost:5555/v2/_catalog
{"repositories":["fraud-detector"]}

root@controlplane ~/code/ml-registry via 🐍 v3.12.3 ➜  curl http://localhost:5555/v2/fraud-detector/tags/list
{"name":"fraud-detector","tags":["v1"]}

```


### Day 55
> Fix a Broken Dockerfile HEALTHCHECK and EXPOSE

```Dockerfile
EXPOSE 8085

HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8085/health')" || exit 1

```

```shell
docker build -t ml-health:v1 .

docker inspect --format '{{.Config.ExposedPorts}}' ml-health:v1

docker run ml-health:v1

```

### Day 56
> Fix a Docker CI Pipeline with Git-SHA Tagging

```shell 
# build.sh
REGISTRY="localhost:5555"


# --- Stage 1: test
python3 -m pytest app/test_app.py


# --- Stage 3: tag with short git SHA
SHA=$(git -C app rev-parse --short HEAD)
TAGGED="$REGISTRY/$IMAGE:$SHA"
```


### Day 57
> Serve an ML Model with Flask

```python
# appy.py

payload = request.get_json() or {}
amount = float(payload.get("amount", 0.0))
hour = int(payload.get("hour", 0))
num_tx_past_day = int(payload.get("num_tx_past_day", 0))
features = np.array([[amount, hour, num_tx_past_day]])
is_fraud = int(MODEL.predict(features)[0])
return jsonify({"is_fraud": is_fraud}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8085)
```



### Day 58
> Serve an ML Model with FastAPI

```python
# TODO 1
amount: float
hour: int = Field(ge=0, le=23)
num_tx_past_day: int = Field(ge=0)


# TODO 2
feature = [[req.amount, req.hour, req.num_tx_past_day]]
is_fraud = int(MODEL.predict(feature)[0])
prediction_history.append({
    "amount": req.amount, 
    "hour": req.hour, 
    "num_tx_past_day": req.num_tx_past_day,
    "is_fraud": is_fraud
})
return PredictResponse(is_fraud=is_fraud)
```

```shell
# To run the fastAPI server
uvicorn app:app --host 0.0.0.0 --port 8085
```

### Day 59
> Run Batch Predictions on a Dataset

```python
model = joblib.load(MODEL_PATH)

df = pd.read_csv(INPUT_CSV)
features = df[["amount", "hour", "num_tx_past_day"]]

df["prediction"] = model.predict(features)

df.to_csv(OUTPUT_CSV,index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")

```

### Day 60
> Package a Model as a BentoML Service

```python
features = np.array([[amount, hour, num_tx_past_day]])
is_fraud = int(self.model.predict(features)[0])
self._history.append({
    "amount": amount, 
    "hour": hour, 
    "num_tx_past_day": num_tx_past_day,
    "is_fraud": is_fraud
})
return {"is_fraud": is_fraud}
```

```shell
bentoml serve service:FraudService
```

### Day 61 (Need practice)
> Deploy a Model-Serving Container via Portainer

`DO it from the UI`






### Day 62
> Implement A/B Testing for Model Deployment

```python
    if random.random() < 0.8 :
        model = MODEL_V1
        version ="v1"
    else :
        model = MODEL_V2
        version ="v2"

    is_fraud = int(model.predict(features)[0])
    return jsonify({"is_fraud": is_fraud, "model_version": version}), 200
```

```shell
# for verification
curl -X POST http://localhost:8085/predict \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 250.75,
       "hour": 14,
       "num_tx_past_day": 3
     }'
```


### Day 63
> Async Predictions with a Redis-Backed Worker

```python
# TODO 1
REDIS.set(RESULT_KEY.format(task_id=task_id), is_fraud, ex=RESULT_TTL_SECONDS)
```
```python
# TODO 2
stored = REDIS.get(RESULT_KEY.format(task_id=task_id))
if stored is None:
    return jsonify({"task_id": task_id, "status": "pending"}), 202
return jsonify({"task_id": task_id, "is_fraud": int(stored)}), 200
```

```shell
curl -X POST http://localhost:8085/predict-async \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 250.75,
       "hour": 14,
       "num_tx_past_day": 3
     }'
```





### Day 64 - (good one)
> Serve Multiple Models Behind Unified API Gateway

```yaml
# docker-compose.yaml
recommend:
  build: ./recommend
  container_name: mm-recommend
```

```json
upstream recommend_backend {
    server recommend:5000;
}
.
.
.
location /recommend/ {
    proxy_pass http://recommend_backend/;
}

```

```shell
# for validation
curl -X POST http://localhost:8085/fraud/predict \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 250.75,
       "hour": 14,
       "num_tx_past_day": 3
     }'


curl -X POST http://localhost:8085/churn/predict \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_days": 75,
       "support_ticket": 14
     }'

curl -X POST http://localhost:8085/recommend/predict \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": 75
     }'
```


### Day 65
> Simulate a Canary Rollout for Model Updates


```python

ROLLBACK_THRESHOLD = 0.05

if self.phase == 1 :
    self.v1_weight=0.95
    self.v2_weight=0.05
elif self.phase == 2 :
    self.v1_weight=0.70
    self.v2_weight=0.30
elif self.phase == 3 :
    self.v1_weight=0.00
    self.v2_weight=1.00
```




### Day 66
> Production Model Serving with Docker Compose


docker-compose.yaml


```python
app = Flask(__name__)
metrics=PrometheusMetrics(app)
```

```yaml
# prometheus.yaml
scrape_configs:
  - job_name: model-api
    static_configs:
      - targets:
          - model-api:5000

```


```json
// nginx.conf
http {
    upstream model_backend {
        server model-api:5000;
    }
```

```shell
docker compose up -d
docker ps
```

`create a dashboard in grafana & save `


```shell
# to validate
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:5000/metrics

curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"amount":3200,"hour":23,"num_tx_past_day":5}' \
  http://localhost:8085/predict

curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | head -30

curl -u admin:grafana2026 http://localhost:3000/api/datasources

curl -u admin:grafana2026 http://localhost:3000/api/search?type=dash-db

```




### Day 67
> Add Prometheus as a Grafana Data Source


- Add datasource 'prometheus` from the UI
- give `http://prometheus:9090` as URL
- Create a Dashboard with prometheus. (It can be anything)


### Day 68
> Build a Grafana Time-Series Panel for Prediction Accuracy

- Create a Dashboard using UI 


### Day 69
> Build a Grafana Table Panel for Per-Feature Data Drift

- Create a Dashboard using UI 


### Day 70 (Hard)
> Enforce Accuracy Gates with an Evidently Test Suite and a Grafana Alert

```python
METRICS.append(DatasetMissingValueCount(tests=[lt(10)]))
METRICS.append(Accuracy(tests=[gt(0.80)]))
```

- create the alert from the UI
- avg_over_time(prediction_accuracy[1m]) < 0.80


```shell
curl -s -u admin:grafana2026 http://localhost:3000/api/v1/provisioning/alert-rules \
  | python3 -m json.tool | head -80
```

### Day 71
> Build a 4-Panel Model-Overview Grafana Dashboard

- Create 4 dashboards from the UI
- 2 timesseries , 1 stat , 1 bar gauge


### Day 72
> Configure a Grafana Contact Point and Notification Policy


- Create Contact point from the UI
- Create Notification policiy by adding route in UI (bit tricky)

```shell
# Validation
curl -s -u admin:grafana2026 http://localhost:3000/api/v1/provisioning/contact-points \
  | python3 -m json.tool

curl -s -u admin:grafana2026 http://localhost:3000/api/v1/provisioning/policies \
  | python3 -m json.tool
```


### Day 73
> Promote a Retrained Model via a Champion/Challenger Gate

```python
champion_version = client.get_model_version_by_alias(MODEL, PROD_ALIAS)
champion = f1_of(champion_version.version)
challenger = f1_of(CHALLENGER_VERSION)

if champion < challenger:
    client.set_registered_model_alias(MODEL, PROD_ALIAS, CHALLENGER_VERSION)
else:
    print("The challenger was rejected")
```

```shell
# Validate
python promote.py
```

### Day 74 (Hard)
> Add a Custom Business Metric and a Grafana Version Variable
```python
FRAUD_AMOUNT = Counter(
    "fraud_amount_usd_total",
    "Total Fraud amount captured",
    labelnames=["version"],
    registry=REGISTRY,
)

# line 71
FRAUD_AMOUNT.labels(version=version).inc(random.uniform(50, 500))
```

```shell
docker compose restart metric-emitter



curl -s 'http://localhost:9090/api/v1/query?query=fraud_amount_usd_total' \
  | python3 -m json.tool

```

- Create a dashboard , but before that create a variable




### Day 75 (Hard)
> Fix and Complete an End-to-End Monitoring Stack: Prometheus, Grafana, Evidently

```yaml
# prometheus.yaml
# Change to 5000 port
scrape_configs:
  - job_name: metric-emitter
    static_configs:
      - targets:
          - metric-emitter:5000
```

```python
# app/metrics-emitter.py
# changed to /metrics
@app.route("/metrics")
def metrics():
    return generate_latest(REGISTRY), 200, {"Content-Type": CONTENT_TYPE_LATEST}
```

```yaml
# /grafana/promethus.yaml
# fixed the port to 9090
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

```shell
docker compose restart

curl -v http://localhost:5000/metrics

curl http://localhost:9090/api/v1/targets  | python3 -m json.tool

curl -s -u admin:grafana2026 http://localhost:3000/api/datasources \
  | python3 -m json.tool
```


- Create 3 panels in a dashboard
- Create tags for the dashboard

```shell
# verify the dashboard creation
DASH_UID=$(curl -s -u admin:grafana2026 'http://localhost:3000/api/search?type=dash-db' \
  | python3 -c "import json, sys; print(json.load(sys.stdin)[0]['uid'])")
curl -s -u admin:grafana2026 http://localhost:3000/api/dashboards/uid/$DASH_UID \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)['dashboard']
print('title:', d.get('title'))
print('tags: ', d.get('tags'))
for p in d.get('panels', []):
    expr = (p['targets'][0].get('expr') if p.get('targets') else '-')
    print(f\"{p['type']:12s} | {p['title']:30s} | {expr}\")
"
```



### Day 76
> Create CI Pipeline for ML Code Linting and Testing

- rename `ci.yaml.template` to `ci.yml`

```yaml
- name: Run ruff
        run: ruff check src tests # TODO: lint `src` and `tests` with ruff

- name: Run pytest
        run: python3 -m pytest tests -v # TODO: run pytest on the tests/ directory with verbose output
```

```shell
git checkout -b add-ci
git add .
git commit -m "Updated workflow"
git push origin add-ci
```
- Create PR & merge it




### Day 77
> Fix a Failing Data-Quality Job in Gitea Actions

```yaml
# Correct the .py file
data-quality:
  steps:
      run: python3 -m pytest tests/test_data_quality.py -v
```

```shell
git add .
git commit -m "Updated workflow"
git push origin add-data-validation
```


### Day 78
> Parallelise Tests via a Gitea Actions Matrix Strategy

```yaml
# Add the Strategy & update the run script
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      suite: [train, data_quality, model_contract]
  steps:
    - name: Run all tests
      run: python3 -m pytest tests/test_${{ matrix.suite }}.py -v
```



### Day 79
> Publish CI Training Artefacts via upload-artifact

```yaml
- name: Upload training artefacts
  uses: actions/upload-artifact@v3
  with:
    name: model-report
    path: artifacts/
```

### Day 80
> Wire Repository Secrets into a Gitea Actions Workflow

- Create Action secrets from the gitea UI

```yaml
  register:
    runs-on: ubuntu-latest
    env: 
      MLFLOW_TRACKING_URI: {{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TOKEN: {{ secrets.MLFLOW_TOKEN }}
```

- Create PR & Merge

```shell
TOKEN=$(cat /root/.gitea/token)

curl -s -H "Authorization: token $TOKEN" \
  http://localhost:3000/api/v1/repos/gitea-admin/fraud-detector/actions/secrets \
  | python3 -m json.tool

curl -s http://localhost:5000/api/2.0/mlflow/registered-models/get?name=fraud-detector \
  | python3 -m json.tool | head -30

```




### Day 81
> Tag a Release and Publish to the Gitea Package Registry

```yaml
- name: Build image
  run: |
    echo "TODO 1: build and tag the image for the registry"
    docker build -t $REGISTRY/$IMAGE:${{ steps.version.outputs.VERSION }} .

# TODO 2: Push the image tagged in TODO 1 to the Gitea container
# registry so it lands under the repo's Packages.
- name: Push image to Gitea registry
  run: |
    echo "TODO 2: publish the tagged image to the registry"
    docker push "$REGISTRY/$IMAGE:${{ steps.version.outputs.VERSION }}"
```

```shell
git add .
git commit -m "updated workflow"
git push
git tag v0.1.0
git push origin v0.1.0
```

### Day 82
> Compose Gitea Workflows via workflow_call


```yaml
jobs:
  lint:
    uses: ./.gitea/workflows/lint.yml

  test:
    uses: ./.gitea/workflows/test.yml

  report:
    uses: ./.gitea/workflows/report.yml
```
- Commit & merge 


### Day 83
> Revert a Broken ML Release via the Gitea Revert Button


