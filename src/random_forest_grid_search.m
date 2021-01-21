% load the data
train = readtable('../data/train_sampled.csv');
X_train = removevars(train, {'Revenue'});
y_train = train{:, {'Revenue'}};

test = readtable('../data/test.csv');
X_test = removevars(test, {'Revenue'});
y_test = test{:, {'Revenue'}};

% all of the configurations for grid search
max_num_split = [2 10 50 100];
num_var_samp = [5, 10, 25, 50, 75];
min_leaf_size_options = [2 10 20];
min_parent_size_options = [2 10 50];
method_options = {'Bag', 'RUSBoost', 'RobustBoost'};
num_trees_options = [10 50 100 500 1000];

% calculate the total number of possibilities 
% in order to gauge progress as grid search is running
total_combos = (length(max_num_split) ...
    * length(min_parent_size_options) ...
    * length(num_var_samp) ...
    * length(min_leaf_size_options) ...
    * length(method_options) ...
    * length(num_trees_options));

results = [];
iterations = 0;

% grid search across all posisble configurations
% save down all of the results and hyperparameters for each loop
for max_split = 1:length(max_num_split)
        for num_var = 1:length(num_var_samp)
            for min_parent_size = 1:length(min_parent_size_options)
                for min_leaf_size = 1:length(min_leaf_size_options)
                    for met_options = 1:length(method_options)
                        for num_tree = 1:length(num_trees_options)
                            
                            iterations = iterations + 1;
                            
                            pct_complete = round(iterations / total_combos * 100, 4);
                            disp(pct_complete)
                            
                            template = templateTree('MaxNumSplits', max_num_split(max_split), ...
                                'MinLeafSize', min_leaf_size_options(min_leaf_size), ...
                                'MinParentSize', min_parent_size_options(min_parent_size), ...
                                'NumVariablesToSample', num_var_samp(num_var));
                            
                            % train the model
                            t_start = tic;
                            mdl = fitcensemble(X_train, y_train, 'Method', char(method_options(met_options)), ...
                                'KFold', 10, 'NumLearningCycles', num_trees_options(num_tree), ...
                                'CrossVal', 'on', 'Learners', template);
                            
                            t_elapsed = toc(t_start);
                            
                            [best_loss, best_ind] = min(kfoldLoss(mdl, 'mode', 'individual'));
                            
                            % Save the best model to a variable
                            best_rf_trained = mdl.Trained{best_ind};
                            
                            % calculate the training and testing metrics
                            training_accuracy = 1 - loss(best_rf_trained, X_train, y_train);
                            
                            [pred_labels, scores]  = predict(best_rf_trained,X_test);
                            testing_accuracy = 1 - loss(best_rf_trained, X_test, y_test);
                            
                            tp = sum((pred_labels == 1) & (y_test == 1));
                            fp = sum((pred_labels == 1) & (y_test == 0));
                            fn = sum((pred_labels == 0) & (y_test == 1));
                    
                            
                            precision = tp / (tp + fp);
                            recall = tp / (tp + fn);
                            F1 = (2 * precision * recall) / (precision + recall);
                            
                            [X,Y,T,AUC,OPTROCPT,suby,subnames] = perfcurve(y_test,scores(:, 2), 1);
                            
                            % put all of the results and hyperparams in to
                            % an array 
                            tmp_results = struct('MaxNumSplits', max_num_split(max_split), ...
                                'MinLeafSize', min_leaf_size_options(min_leaf_size), ...
                                'MinParentSize', min_parent_size_options(min_parent_size), ...
                                'NumVariablesToSample', num_var_samp(num_var), ...
                                'NumLearningCycles', num_trees_options(num_tree), ...
                                'Method', char(method_options(met_options)), ...
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
                            % append it to the results array
                            results = [results, tmp_results];
                            
                            if(mod(iterations, 10) == 0)
                                save('rf_results_smote_numtrees.mat', 'results')
                            end
                        end
                    end
                end
            end
        end
end
% save after everything is complete
save('rf_results_smote_numtrees.mat', 'results')


