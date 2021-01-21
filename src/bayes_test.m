% load in the data and split it in to features and target
test = readtable('test.csv');
X_test = removevars(test, {'Revenue'});
y_test = test{:, {'Revenue'}};

% load in the best model
load best_nb_trained

% calculate predictions and testing metrics
[pred_labels, scores] = predict(best_nb_trained,X_test);
test_accuracy = sum(pred_labels == y_test) / numel(y_test) * 100;

% plot confusion chart from within matlab
confusionchart(pred_labels,y_test);

tp = sum((pred_labels == 1) & (y_test == 1));
fp = sum((pred_labels == 1) & (y_test == 0));
fn = sum((pred_labels == 0) & (y_test == 1));

precision = tp / (tp + fp);
recall = tp / (tp + fn);
F1 = (2 * precision * recall) / (precision + recall);

[X,Y,T,AUC,OPTROCPT,suby,subnames] = perfcurve(y_test,scores(:, 2), 1);