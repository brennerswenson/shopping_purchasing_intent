% load the data
train = readtable('train_sampled.csv');
X_train = removevars(train, {'Revenue'});
y_train = train{:, {'Revenue'}};

tic
% train the model using hyperparameters found via grid search
template = templateTree('MaxNumSplits', 100, 'NumVariablesToSample', 75, ...
                        'MinLeafSize', 10, 'MinParentSize', 10, ... 
                        'Prune', 'Off', 'Surrogate', 'On');
mdl = fitcensemble(X_train,y_train, 'Method', 'Bag', ...
    'Learners', template, 'CrossVal', 'on', 'NumLearningCycles', 50);

[best_loss, best_ind] = min(kfoldLoss(mdl, 'mode', 'individual'));
% Save the best model to a variable
best_rf_trained = mdl.Trained{best_ind};
toc

% Save model for future use
save('best_rf_trained','best_rf_trained')