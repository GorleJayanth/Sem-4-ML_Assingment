def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred_pipeline = pipeline.predict(X_test)
    model_type = 'classification' if isinstance(pipeline.named_steps['stacking'], StackingClassifier) else 'regression'

    print(f'Pipeline Score: {pipeline.score(X_test, y_test):.4f}')
    if model_type == 'classification':
        print("Classification Report (Pipeline):")
        print(classification_report(y_test, y_pred_pipeline))
    else:
        print("Regression Metrics (Pipeline):")
        print(f'MSE: {mean_squared_error(y_test, y_pred_pipeline):.4f}')
        print(f'R² Score: {r2_score(y_test, y_pred_pipeline):.4f}')

    stacking_model = pipeline.named_steps['stacking']
    scaler = pipeline.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stacking_model.fit(X_train_scaled, y_train)
    y_pred_stacking = stacking_model.predict(X_test_scaled)

    print(f'Stacking Model Score: {stacking_model.score(X_test_scaled, y_test):.4f}')
    if model_type == 'classification':
        print("Classification Report (Stacking Classifier):")
        print(classification_report(y_test, y_pred_stacking))
    else:
        print("Regression Metrics (Stacking Regressor):")
        print(f'MSE: {mean_squared_error(y_test, y_pred_stacking):.4f}')
        print(f'R² Score: {r2_score(y_test, y_pred_stacking):.4f}')

    return pipeline
