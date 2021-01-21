% load the data
train = readtable('train_sampled.csv');
X_train = removevars(train, {'Revenue'});
y_train = train{:, {'Revenue'}};

% train the model using optimised hyperparameters
tic
mdl = fitcnb(X_train, y_train, ...
         'DistributionNames', 'kernel', ...
         'Width', 0.05, ...
         'Kernel', 'Triangle', ...
         'KFold', 20, ...
         'CrossVal', 'on');

 % extract the model with the best performance from k-folds
[best_loss, best_ind] = min(kfoldLoss(mdl, 'mode', 'individual'));

% Save the best model to a variable
best_nb_trained = mdl.Trained{best_ind};
toc

% Save model for future use
save('best_nb_trained','best_nb_trained')