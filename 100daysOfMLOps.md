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


### Day 45
> Authenticate MLflow to Vault via AppRole and Fix Its KV Policy



