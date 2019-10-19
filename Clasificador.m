%% Dictionary
% Binary classification: SVM->fitcsvm

% Multiclass classification: SVM->fitcecoc ; Neural Networks->feedforwardnet

% Classifier Performance: confusion matrix -> Confusiomat ; Data partition
% -> cvpartition ; Cross validation -> crossval

% +------------------------------+---------------------------------------+----------------------+-----+
% |                              | C1                                    | C2                   |     |
% | Actual class\Predicted class |                                       |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% |                              | True positives (TP)                   |                      |     |
% | C1                           |                                       | False negatives (FN) | P   |
% +------------------------------+---------------------------------------+----------------------+-----+
% | C2                           | False positives (FP)                  | True negatives (TN)  |     |
% |                              |                                       |                      | N   |
% +------------------------------+---------------------------------------+----------------------+-----+
% |                              | P'                                    | N'                   | All |
% +------------------------------+---------------------------------------+----------------------+-----+


% +------------------------------+---------------------------------------+
% | Accuracy                     | (TP+TN)/ALL                           |
% +------------------------------+---------------------------------------+
% | Error rate                   | (FP+FN)/ALL                           |
% +------------------------------+---------------------------------------+
% | Sensitivity                  | TP/P                                  |
% +------------------------------+---------------------------------------+
% | Specificity                  | TN/N                                  |
% +------------------------------+---------------------------------------+
% | Precision                    | TP/(TP+FP)                            |
% +------------------------------+---------------------------------------+
% | Recall                       | TP/(TP+FP)                            |
% +------------------------------+---------------------------------------+
% | F-score                      | (2*precision*recall)/precision+recall |
% +------------------------------+---------------------------------------+