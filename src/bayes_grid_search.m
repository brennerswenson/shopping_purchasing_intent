% load the data
train = readtable('train_sampled.csv');
X_train = removevars(train, {'Revenue'});
y_train = train{:, {'Revenue'}};

test = readtable('test.csv');
X_test = removevars(test, {'Revenue'});
y_test = test{:, {'Revenue'}};
        
% all of the grid search options
width_opt = [0.005 0.01 0.05 0.1 1 5 10 15];
kernel_ops = {'box', 'epanechnikov', 'normal', 'triangle'};
k_fold_options = [5 10 20 25];

% find the total number of possible configurations
total_combos = (length(width_opt) ...
    * length(kernel_ops) ...
    * length(k_fold_options));

% empty array to store results
results = [];
iterations = 0;

% grid search
% iterate through every possible configuration 
% and train a model while saving down the 
% params used in addition to performance metrics
for width = 1:length(width_opt)
    for kernel = 1:length(kernel_ops)
        for k_fold = 1:length(k_fold_options)
            
            iterations = iterations + 1;

            pct_complete = round(iterations / total_combos * 100, 4);
            disp(pct_complete)
            
            % fit the model
            t_start = tic;
            mdl = fitcnb(X_train, y_train, ...
                         'DistributionNames', 'kernel', ...
                         'Width', width_opt(width), ...
                         'Kernel', char(kernel_ops(kernel)), ...
                         'KFold', k_fold_options(k_fold), ...
                         'CrossVal', 'on');
            t_elapsed = toc(t_start);
            [best_loss, best_ind] = min(kfoldLoss(mdl, 'mode', 'individual'));


            % Save the best model to a variable
            best_nb_trained = mdl.Trained{best_ind};
            
            % calculate metrics
            training_accuracy = 1 - loss(best_nb_trained, X_train, y_train);

            [pred_labels, scores]  = predict(best_nb_trained, X_test);
            testing_accuracy = 1 - loss(best_nb_trained, X_test, y_test);

            tp = sum((pred_labels == 1) & (y_test == 1));
            fp = sum((pred_labels == 1) & (y_test == 0));
            fn = sum((pred_labels == 0) & (y_test == 1));

            precision = tp / (tp + fp);
            recall = tp / (tp + fn);
            F1 = (2 * precision * recall) / (precision + recall);

            [X,Y,T,AUC,OPTROCPT,suby,subnames] = perfcurve(y_test,scores(:, 2), 1);
            
            % package all results and hyperparams into struct
            tmp_results = struct('Width', width_opt(width), ...
            'Kernel', char(kernel_ops(kernel)), ...
            'KFold', k_fold_options(k_fold), ...
            'TrainingAccuracy', training_accuracy, ...
            'TestingAccuracy', testing_accuracy, ...
            'Precision', precision, ...
            'Recall', recall, ...
            'F1', F1, ...
            'X', X, ...
            'Y', Y, ...
            'T', T, ...
            'AUC', AUC, ...
            'OPTROCPT', OPTROCPT, ...
            'suby', suby, ...
            'subnames', subnames, ...
            'TrainingTime', t_elapsed);

        
            % append to results array
            results = [results, tmp_results];
            
            % save down every 10 times
            if(mod(iterations, 10) == 0)
                save('nb_results.mat', 'results')
            
            end
        end
    end
end
% save after everything is done
save('nb_results.mat', 'results')
