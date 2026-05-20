import mlflow
mlflow.set_tracking_uri("http://34.163.157.122:5000/")

# Dernier champion SFT
exp_sft = mlflow.get_experiment_by_name("sft-qwen3-medical")
runs_sft = mlflow.search_runs(
    experiment_ids=[exp_sft.experiment_id],
    filter_string='tags.model_status = "champion" and tags.stage = "sft"',
    order_by=["start_time DESC"], max_results=3,
)
print("=== Champions SFT ===")
print(runs_sft[["run_id", "start_time", "end_time"]].to_string())

# Dernier champion DPO
exp_dpo = mlflow.get_experiment_by_name("sft-qwen3-medical")
runs_dpo = mlflow.search_runs(
    experiment_ids=[exp_dpo.experiment_id],
    filter_string='tags.model_status = "champion" and tags.stage = "dpo"',
    order_by=["start_time DESC"], max_results=3,
)
print("\n=== Champions DPO ===")
print(runs_dpo[["run_id", "start_time", "end_time"]].to_string())