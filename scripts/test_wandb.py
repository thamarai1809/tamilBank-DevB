import wandb

wandb.init(project="tamilbank", name="g2p-sanity-check")

# pretend metric for now, just to confirm logging works
wandb.log({"metric_name": 0.85})
wandb.log({"metric_name": 0.90})
wandb.log({"metric_name": 0.92})

wandb.finish()