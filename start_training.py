# start_training.py
import os, time
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput

# ---- CONFIGURE THESE ----
ROLE_ARN = os.getenv("SM_ROLE_ARN")            # set via GitHub Actions secret
BUCKET   = "prathvi-raw"                       # your bucket
PROCESSED_PREFIX = "processed/"                # input data
OUTPUT_S3 = f"s3://{BUCKET}/models/"           # where model artifacts go
FRAMEWORK_VERSION = "1.2-1"                    # scikit-learn container version
INSTANCE_TYPE = "ml.m5.large"                  # smallest common training type
# -------------------------

session = sagemaker.Session()
region = session.boto_region_name
account = sagemaker.session.get_caller_identity_arn().split(":")[4]

# Estimator (Script Mode runs train.py that you already added)
estimator = SKLearn(
    entry_point="train.py",
    role=ROLE_ARN,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    framework_version=FRAMEWORK_VERSION,
    py_version="py3",
    output_path=OUTPUT_S3,
    sagemaker_session=session,
    base_job_name="etl-ml-diabetes"
)

# Training input points to processed CSVs
train_input = TrainingInput(
    s3_data=f"s3://{BUCKET}/{PROCESSED_PREFIX}",
    content_type="text/csv"
)

print("Starting training job…")
estimator.fit({"train": train_input}, wait=True, logs=True)
print("Training complete.")
print(f"Model artifact at: {estimator.model_data}")
