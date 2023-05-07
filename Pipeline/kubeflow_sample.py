import kfp
from kfp import dsl
from kfp.components import func_to_container_op

@func_to_container_op
def read_data(data_url):
    import pandas as pd

    data = pd.read_csv(data_url)
    return data.to_json()

@func_to_container_op
def preprocess_data(data_json):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    data = pd.read_json(data_json)
    
    cat_vars = data.select_dtypes(include=['object']).columns
    num_vars = data.select_dtypes(exclude=['object']).columns
    encoder = LabelEncoder()
    encoded_vars = data[cat_vars].apply(encoder.fit_transform)
    data_encoded = pd.concat([data[num_vars], encoded_vars], axis=1)

    return data_encoded.to_json()

@func_to_container_op
def split_data(data_encoded_json):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_encoded = pd.read_json(data_encoded_json)
    target = 'isFlaggedFraud'
    X = data_encoded.drop(target, axis=1)
    y = data_encoded[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train.to_json(), X_test.to_json(), y_train.to_json(), y_test.to_json()

@func_to_container_op
def train_single_model(X_train_json, X_test_json, y_train_json, y_test_json, penalty, C):
    import json
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_train = pd.read_json(X_train_json)
    X_test = pd.read_json(X_test_json)
    y_train = pd.read_json(y_train_json, typ='series')
    y_test = pd.read_json(y_test_json, typ='series')

    model = LogisticRegression(penalty=penalty, C=C, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return json.dumps({"model_params": {"penalty": penalty, "C": C}, "accuracy": accuracy})

@func_to_container_op
def select_best_model(*model_results_json):
    import json

    accuracies = []
    model_params = []
    for model_result_json in model_results_json:
        model_result = json.loads(model_result_json)
        accuracies.append(model_result["accuracy"])
        model_params.append(model_result["model_params"])

    best_index = accuracies.index(max(accuracies))
    best_model_params = model_params[best_index]
    best_accuracy = accuracies[best_index]

    return json.dumps({"best_model": best_model_params, "best_accuracy": best_accuracy})


@dsl.pipeline(name="ModelPipeline", description="A pipeline")
def model_pipeline(data_url):
    data_op = read_data(data_url)
    preprocessed_data_op = preprocess_data(data_op.output)
    split_data_op = split_data(preprocessed_data_op.output)

    model_params = [
        {'penalty': 'l1', 'C': 1},
        {'penalty': 'l2', 'C': 1},
        {'penalty': 'l1', 'C': 0.5},
        {'penalty': 'l2', 'C': 0.5}
    ]

    train_single_model_ops = []
    with dsl.ParallelFor(model_params) as item:
        train_single_model_op = train_single_model(
            split_data_op.outputs["output_0"],
            split_data_op.outputs["output_1"],
            split_data_op.outputs["output_2"],
            split_data_op.outputs["output_3"],
            item.penalty,
            item.C
        )
        train_single_model_ops.append(train_single_model_op)

    select_best_model_op = select_best_model(*[op.output for op in train_single_model_ops])

pipeline_func = model_pipeline("credit_filtered.csv")

