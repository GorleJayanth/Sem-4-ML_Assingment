def explain_model(pipeline, X_train, X_test, y_train, y_test, feature_names):
    model_type = 'classification' if isinstance(pipeline.named_steps['stacking'], StackingClassifier) else 'regression'

    explainer = LimeTabularExplainer(
        X_train,
        mode=model_type,
        feature_names=feature_names,
        discretize_continuous=True
    )

    instance = X_test[0]
    true_label = y_test[0]

    prediction_function = (
        pipeline.predict_proba if model_type == 'classification' else pipeline.predict
    )

    prediction = prediction_function([instance])
    predicted_class = np.argmax(prediction[0]) if model_type == 'classification' else prediction[0]

    print("\nLIME Explanation for 1st Test Instance:")
    print(f"Actual Class: {true_label}")
    print(f"Predicted Class: {predicted_class}")
    if model_type == 'classification':
        print(f"Predicted Probabilities: {np.round(prediction[0], 4)}")

    exp = explainer.explain_instance(instance, prediction_function)

    try:
        exp.show_in_notebook()
    except:
        print("\nFeature Contributions (LIME):")
        print(exp.as_list())

file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
X, y, feature_names = load_data(file_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = create_pipeline('classification')
trained_pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

explain_model(trained_pipeline, X_train, X_test, y_train, y_test, feature_names)
