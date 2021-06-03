from google.cloud import automl
from google.oauth2 import service_account
from google.cloud import error_reporting

dataset_id = 'TCN4939068354376761344'
project_id = 'disco-plane-308520'
path = 'gs://disco-plane-308520-lcm/test.csv'
region = 'us-central1'

credentials = service_account.Credentials.from_service_account_file("disco-plane-308520-c946f9c07b95.json")
client = automl.AutoMlClient(credentials=credentials)
error_client = error_reporting.Client(credentials=credentials)

# Get the full path of the dataset.
dataset_full_id = client.dataset_path(project_id, region, dataset_id)
# Get the multiple Google Cloud Storage URIs
input_uris = path.split(",")
gcs_source = automl.GcsSource(input_uris=input_uris)
input_config = automl.InputConfig(gcs_source=gcs_source)
# Import data from the input URI


response = client.import_data(name=dataset_full_id, input_config=input_config)

print("Processing import...")

try:
    print("Data imported. {}".format(response.result()))
except Exception:
    error_client.report_exception()