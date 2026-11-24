from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_code, flow_evaluate, flow_predict, flow_train

TZ = "Africa/Tunis"

if __name__ == "__main__":
    all_deploy = flow_all.to_deployment(
        name="ml-pipeline-all",
        schedules=[CronSchedule(cron="0 2 * * *", timezone=TZ)],
        tags=["mlops", "full-pipeline"],
    )
    train_deploy = flow_train.to_deployment(
        name="ml-pipeline-train",
        schedules=[CronSchedule(cron="0 3 * * 1", timezone=TZ)],
        tags=["mlops", "train"],
    )
    predict_deploy = flow_predict.to_deployment(
        name="ml-pipeline-predict",
        schedules=[CronSchedule(cron="0 * * * *", timezone=TZ)],
        tags=["mlops", "predict"],
    )
    evaluate_deploy = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["mlops", "evaluate", "manuel"],
    )
    code_deploy = flow_code.to_deployment(
        name="ml-pipeline-code",
        tags=["mlops", "code", "manuel"],
    )
    serve(all_deploy, train_deploy, predict_deploy, evaluate_deploy, code_deploy)
