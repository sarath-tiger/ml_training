## src folder consists of the python script

## Without parameters
## Test Case 1: To run the test/unit_test script go to the respective folder and execute the below commands
python test_data_ingestion.py

## Test Case 2: To run the test/functional_test script go to the respective folder and execute the below commands
python test_training.py

## With parameters
## Test case 4 : To run test/unit_test script go to the respective folder and execute the below commands
python test_data_ingestion.py $dataset_out_path $split_data_path

## Test case 3: To run the test/functional_test script go to the respective folder and execute the below commands
python test_training.py $split_data_path $ml_model_path
