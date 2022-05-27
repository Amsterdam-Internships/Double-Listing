import pandas as pd
import numpy as np
import abc
import sys
import warnings
from typing import Union, Callable, Optional, Tuple, List, Iterator, Any
import numpy as np
from sklearn.base import BaseEstimator
from modAL.models.base import BaseLearner
import scipy.sparse as sp
from modAL.utils.data import data_vstack, data_hstack, modALinput, retrieve_rows
import numpy as np
from typing import Callable, Optional, Tuple, List, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from modAL.utils.validation import check_class_labels, check_class_proba
from modAL.utils.data import modALinput, retrieve_rows
from collections import Counter
from typing import Tuple
from sklearn.base import BaseEstimator
from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax,shuffled_argmax
from modAL.disagreement import vote_entropy
from modAL.models import ActiveLearner

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})
def to_bin(boolean):
    if boolean == True:
        return 1
    else:
        return 0
def compare(df,id_1,id_2,full =False):
    link = [x for x in df['listing_url'][df['ListingId']==id_1]]
    link_2 = [x for x in df['listing_url'][df['ListingId']==id_2]]
    if full == True:
    #Compare URLS
        lista = [print(x,':',y,'==',z) for x,y,z in zip(df.columns.tolist(),
                                                        df[df['ListingId'] == id_1].values.tolist()[0],
                                                        df[df['ListingId'] == id_2].values.tolist()[0])]
        return 'Full info retrieved'
    else:
        print(link[0])
        print(link_2[0])
        return 'Links retreived'
class BaseCommittee(ABC, BaseEstimator):

    #Retreived from moDAL Library
    """
    Base class for query-by-committee setup.

    Args:
        learner_list: List of ActiveLearner objects to form committee.
        query_strategy: Function to query labels.
        on_transformed: Whether to transform samples with the pipeline defined by each learner's estimator
            when applying the query strategy.
    """
    def __init__(self, learner_list: List[BaseLearner], query_strategy: Callable, on_transformed: bool = False, boots: bool = False) -> None:
        assert type(learner_list) == list, 'learners must be supplied in a list'

        self.learner_list = learner_list
        self.query_strategy = query_strategy
        self.on_transformed = on_transformed
        self.boots= boots
        # TODO: update training data when using fit() and teach() methods
        self.X_training = None

    def __iter__(self) -> Iterator[BaseLearner]:
        for learner in self.learner_list:
            yield learner

    def __len__(self) -> int:
        return len(self.learner_list)

    def _add_training_data(self, X: modALinput, y: modALinput) -> None:
        """
        Adds the new data and label to the known data for each learner, but does not retrain the model.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.

        Note:
            If the learners have been fitted, the features in X have to agree with the training samples which the
            classifier has seen.
        """
        for learner in self.learner_list:
            learner._add_training_data(X, y)

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> None:
        """
        Fits all learners to the training data and labels provided to it so far.

        Args:
            bootstrap: If True, each estimator is trained on a bootstrapped dataset. Useful when
                using bagging to build the ensemble.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        for learner in self.learner_list:
            learner._fit_to_known(bootstrap=bootstrap, **fit_kwargs)

    def _fit_on_new(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> None:
        """
        Fits all learners to the given data and labels.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, the method trains the model on a set bootstrapped from X.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        for learner in self.learner_list:
            learner._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)

    def fit(self, X: modALinput, y: modALinput, **fit_kwargs) -> 'BaseCommittee':
        """
        Fits every learner to a subset sampled with replacement from X. Calling this method makes the learner forget the
        data it has seen up until this point and replaces it with X! If you would like to perform bootstrapping on each
        learner using the data it has seen, use the method .rebag()!

        Calling this method makes the learner forget the data it has seen up until this point and replaces it with X!

        Args:
            X: The samples to be fitted on.
            y: The corresponding labels.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        for learner in self.learner_list:
            learner.fit(X, y, **fit_kwargs)

        return self

    def transform_without_estimating(self, X: modALinput) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Transforms the data as supplied to each learner's estimator and concatenates transformations.
        Args:
            X: dataset to be transformed

        Returns:
            Transformed data set
        """
        return data_hstack([learner.transform_without_estimating(X) for learner in self.learner_list])

    def query(self, X_pool,boots: bool = False ,*query_args, **query_kwargs) -> Union[Tuple, modALinput]:
        """
        Finds the n_instances most informative point in the data provided by calling the query_strategy function.

        Args:
            X_pool: Pool of unlabeled instances to retrieve most informative instances from
            *query_args: The arguments for the query strategy. For instance, in the case of
                :func:`~modAL.disagreement.max_disagreement_sampling`, it is the pool of samples from which the query.
                strategy should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.

        Returns:
            Return value of the query_strategy function. Should be the indices of the instances from the pool chosen to
            be labelled and the instances themselves. Can be different in other cases, for instance only the instance to
            be labelled upon query synthesis.
        """
        query_result = self.query_strategy(self, X_pool,boots=boots *query_args, **query_kwargs)
        if isinstance(query_result, tuple):
            warnings.warn("Query strategies should no longer return the selected instances, "
                          "this is now handled by the query method. "
                          "Please return only the indices of the selected instances", DeprecationWarning)
            return query_result

        return query_result, retrieve_rows(X_pool, query_result)

    def rebag(self, **fit_kwargs) -> None:
        """
        Refits every learner with a dataset bootstrapped from its training instances. Contrary to .bag(), it bootstraps
        the training data for each learner based on its own examples.

        Todo:
            Where is .bag()?

        Args:
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._fit_to_known(bootstrap=True, **fit_kwargs)

    def teach(self, X: modALinput, y: modALinput, bootstrap: bool = False, only_new: bool = False, **fit_kwargs) -> None:
        """
        Adds X and y to the known training data for each learner and retrains learners with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, trains each learner on a bootstrapped set. Useful when building the ensemble by bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._add_training_data(X, y)
        if not only_new:
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            self._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)

    @abc.abstractmethod
    def predict(self, X: modALinput) -> Any:
        pass

    @abc.abstractmethod
    def vote(self, X: modALinput) -> Any:  # TODO: clarify typing
        pass
def vote_entropy_sampling(committee: BaseCommittee, X: modALinput,
                          n_instances: int = 1, random_tie_break=False,boots:bool = False,
                          **disagreement_measure_kwargs) -> np.ndarray:
    """
    Vote entropy sampling strategy.

    Args:
        committee: The committee for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **disagreement_measure_kwargs: Keyword arguments to be passed for the disagreement
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
         the instances from X chosen to be labelled.
    """
    if boots == False:
        n_instances = 1
    else:
        n_instances =100
    disagreement = vote_entropy(committee, X, **disagreement_measure_kwargs)
    if not random_tie_break:
        return multi_argmax(disagreement, n_instances=n_instances)
    return shuffled_argmax(disagreement, n_instances=n_instances)
class Committee(BaseCommittee):
    """
    This class is an abstract model of a committee-based active learning algorithm.
    """
    def __init__(self, learner_list: List[ActiveLearner], query_strategy: Callable = vote_entropy_sampling,
                 on_transformed: bool = False, boots: bool = False) -> None:
        super().__init__(learner_list, query_strategy, on_transformed,boots)
        self._set_classes()

    def _set_classes(self):
        """
        Checks the known class labels by each learner, merges the labels and returns a mapping which maps the learner's
        classes to the complete label list.
        """
        # assemble the list of known classes from each learner
        try:
            # if estimators are fitted
            known_classes = tuple(learner.estimator.classes_ for learner in self.learner_list)
        except AttributeError:
            # handle unfitted estimators
            self.classes_ = None
            self.n_classes_ = 0
            return

        self.classes_ = np.unique(
            np.concatenate(known_classes, axis=0),
            axis=0
        )
        self.n_classes_ = len(self.classes_)

    def _add_training_data(self, X: modALinput, y: modALinput):
        super()._add_training_data(X, y)

    def fit(self, X: modALinput, y: modALinput, **fit_kwargs) -> 'BaseCommittee':
        """
        Fits every learner to a subset sampled with replacement from X. Calling this method makes the learner forget the
        data it has seen up until this point and replaces it with X! If you would like to perform bootstrapping on each
        learner using the data it has seen, use the method .rebag()!

        Calling this method makes the learner forget the data it has seen up until this point and replaces it with X!

        Args:
            X: The samples to be fitted on.
            y: The corresponding labels.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().fit(X, y, **fit_kwargs)
        self._set_classes()

    def teach(self, X: modALinput, y: modALinput, bootstrap: bool = False, only_new: bool = False, **fit_kwargs) -> None:
        """
        Adds X and y to the known training data for each learner and retrains learners with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, trains each learner on a bootstrapped set. Useful when building the ensemble by bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().teach(X, y, bootstrap=bootstrap, only_new=only_new, **fit_kwargs)
        self._set_classes()

    def predict(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the class of the samples by picking the consensus prediction.

        Args:
            X: The samples to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            The predicted class labels for X.
        """
        # getting average certainties
        proba = self.predict_proba(X, **predict_proba_kwargs)
        # finding the sample-wise max probability
        max_proba_idx = np.argmax(proba, axis=1)
        # translating label indices to labels
        return self.classes_[max_proba_idx]

    def predict_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Consensus probabilities of the Committee.

        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            Class probabilities for X.
        """
        return np.mean(self.vote_proba(X, **predict_proba_kwargs), axis=1)

    def score(self, X: modALinput, y: modALinput, sample_weight: List[float] = None) -> Any:
        """
        Returns the mean accuracy on the given test data and labels.

        Todo:
            Why accuracy?

        Args:
            X: The samples to score.
            y: Ground truth labels corresponding to X.
            sample_weight: Sample weights.

        Returns:
            Mean accuracy of the classifiers.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def vote(self, X: modALinput, **predict_kwargs) -> Any:
        """
        Predicts the labels for the supplied data for each learner in the Committee.

        Args:
            X: The samples to cast votes.
            **predict_kwargs: Keyword arguments to be passed to the :meth:`predict` of the learners.

        Returns:
            The predicted class for each learner in the Committee and each sample in X.
        """
        prediction = np.zeros(shape=(X.shape[0], len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            prediction[:, learner_idx] = learner.predict(X, **predict_kwargs)

        return prediction

    def vote_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the probabilities of the classes for each sample and each learner.

        Args:
            X: The samples for which class probabilities are to be calculated.
            **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the learners.

        Returns:
            Probabilities of each class for each learner and each instance.
        """
        # get dimensions
        n_samples = X.shape[0]
        n_learners = len(self.learner_list)
        proba = np.zeros(shape=(n_samples, n_learners, self.n_classes_))
        # checking if the learners in the Committee know the same set of class labels
        if check_class_labels(*[learner.estimator for learner in self.learner_list]):
            # known class labels are the same for each learner
            # probability prediction is straightforward
            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :] = learner.predict_proba(X, **predict_proba_kwargs)
        else:
            for learner_idx, learner in enumerate(self.learner_list):
                proba[:, learner_idx, :] = check_class_proba(
                    proba=learner.predict_proba(X, **predict_proba_kwargs),
                    known_labels=learner.estimator.classes_,
                    all_labels=self.classes_
                )
        return proba
def al(df,df_test,queries = 20,baseline=False,committee_pred=False,incremental_comitee = False,boostrap = False,warm =False):
    if incremental_comitee == True:
        #Commitee of models
        members = [randomforest_main,desiciontree,logisticreg,xbg_class,svm]
    if incremental_comitee == False:
         members = [randomforest_com,desiciontree,logisticreg,xbg_class,svm]

    #List with comitee object
    learner_list = list()

    if baseline == False:
        #Only use the data corresponding to the features to train
        not_train = ['ListingId_1','ListingId_2','ids', 'agg_score', 'weights','uns_label','label']
    if baseline == True:
        #Only use the data corresponding to the features to train
        not_train = ['source_id','target_id','ids', 'agg_score', 'weights','uns_label','label']

    ids = df['ids'].values
    X_pool = df.drop(not_train, axis=1).values
    y_uns_label = df['uns_label'].values
    y_uns_weight = df['weights'].values

    y_pool_labels = df['label'].values
    y_test = df_test['label'].values
    X_test = df_test.drop(not_train, axis=1).values

    #Will save the new labels being labeled
    new_labels = []

    if boostrap == True:
        # Selecting the 2 most confident labels, a positive a negative with weight 1
        train_idx = np.where(y_uns_weight> 0.99998)
        #This is the labeled pool it starts with 2 examples
        X_train_lb = X_pool[train_idx]
        y_train_lb = y_uns_label[train_idx]
        #Delete used results
        X_pool = np.delete(X_pool, train_idx, axis=0)
        ids = np.delete(ids, train_idx, axis=0)
        y_uns_label = np.delete(y_uns_label, train_idx, axis=0)
        y_uns_weight = np.delete(y_uns_weight, train_idx, axis=0)
        y_pool_labels = np.delete(y_pool_labels, train_idx, axis=0)
    else:

        #If no boostrapping the model is initialized with random instances until there is one positive and negative
        train_idx = np.random.choice(X_pool.shape[0], 1, replace=False)
        X_train_lb = X_pool[train_idx]
        if baseline ==True:
                y_train_lb = y_pool_labels[train_idx]
        else:
            #
            if np.isnan(y_pool_labels[train_idx]):
                print("Are this two listings the same one? 1-match , 0-nonmatch, 2-more=info")
                ids_int = [int(x) for x in ids[train_idx][0][1:-1].split(',')]
                print(compare(df_full,ids_int[0],ids_int[1]))
                label = int(input())
                if label == 2:
                    print(compare(df_full,ids_int[0],ids_int[1],full=True))
                    label = int(input())
                    y_new = np.array([label], dtype=float)
                else:
                    y_new = np.array([label], dtype=float)
                new_labels.append([(ids_int[0],ids_int[1]),label])
                y_train_lb = y_new
            else:
                y_train_lb = y_pool_labels[train_idx]


        #Delete instances moved
        X_pool = np.delete(X_pool, train_idx, axis=0)
        ids = np.delete(ids, train_idx, axis=0)
        y_uns_label = np.delete(y_uns_label, train_idx, axis=0)
        y_uns_weight = np.delete(y_uns_weight, train_idx, axis=0)
        y_pool_labels = np.delete(y_pool_labels, train_idx, axis=0)


        while 1 not in y_train_lb or 0 not in y_train_lb :
            # Keep selecting till at least 1 postive and neagtive label has been added
            train_idx = np.random.choice(X_pool.shape[0], 1, replace=False)
            X_train_lb = np.concatenate((X_train_lb,X_pool[train_idx]),axis=0)
            #For baseline just take the correct label
            if baseline ==True:
                y_train_lb = np.concatenate((y_train_lb,y_pool_labels[train_idx]),axis =0)
            else:
                #If label not available ask for it to the oracle
                if np.isnan(y_pool_labels[train_idx]):
                    print("Are this two listings the same one? 1-match , 0-nonmatch, 2-more=info")
                    ids_int = [int(x) for x in ids[train_idx][0][1:-1].split(',')]
                    print(compare(df_full,ids_int[0],ids_int[1]))
                    label = int(input())
                    #Extra info needed
                    if label == 2:
                        print(compare(df_full,ids_int[0],ids_int[1],full=True))
                        label = int(input())
                        y_new = np.array([label], dtype=float)
                    else:
                        y_new = np.array([label], dtype=float)
                    new_labels.append([(ids_int[0],ids_int[1]),label])
                    y_train_lb = np.concatenate((y_train_lb,y_new),axis =0)
                #If label has already been given in past runs just take it
                else:
                    y_train_lb = np.concatenate((y_train_lb,y_pool_labels[train_idx]),axis =0)

            #Delete from unlabeled pool
            X_pool = np.delete(X_pool, train_idx, axis=0)
            ids = np.delete(ids, train_idx, axis=0)
            y_uns_label = np.delete(y_uns_label, train_idx, axis=0)
            y_uns_weight = np.delete(y_uns_weight, train_idx, axis=0)
            y_pool_labels = np.delete(y_pool_labels, train_idx, axis=0)

        #Check proggress
        print(X_train_lb.shape,y_train_lb.shape)

    #Only if warm adn boostraapp is specified we use the unsupervied labels
    if warm == True and boostrap == True:
        # initializing main random forest
        model_main = randomforest_main()
        model_main.fit(X_pool,y_uns_label,sample_weight= y_uns_weight)
        print('unsupervised')
    #Just Using  the warm true rf but not uns labels
    if warm == True and boostrap == False:
        # initializing main random forest
        model_main = randomforest_main()
        model_main.fit(X_train_lb,y_train_lb)
    #Normal RF
    if warm ==False:
        model_main = randomforest_com()
        model_main.fit(X_train_lb,y_train_lb)

    for clf in members:
        # initializing learner
        learner = ActiveLearner(
            estimator=clf(),
            X_training=X_train_lb, y_training=y_train_lb
            )
        learner_list.append(learner)


    # assembling the committee
    committee = Committee(learner_list=learner_list,query_strategy=vote_entropy_sampling,boots=boostrap)



    # we want to only use the prediction of the random forest which will be incrementally built

    #Get predictions of test set
    if committee_pred == False:
        y_pred = model_main.predict(X_test)
    if committee_pred == True:
        y_pred = committee.predict(X_test)

    #Calculate evaluation metrics
    precision_recall_fscore= precision_recall_fscore_support(y_test,y_pred,average='binary',zero_division=0)
    precision_scores = [precision_recall_fscore[0]]
    recall_scores=  [precision_recall_fscore[1]]
    f_score = [precision_recall_fscore[2]]



    # query by committee
    n_queries = queries
    #Active Learning Loop
    for idx in range(n_queries):
        print('Iteration:',idx)
        # Committee models gives the instance to be labeled
        query_idx, query_instance = committee.query(X_pool)

        if boostrap == True:
            #Get the predictions of the most informative instances
            preds =committee.predict(X_pool)
            #idx_new = np.array([np.argwhere(ids_main == ids[x])[0] for x in query_idx]).squeeze()

            #Only chose the instances which disagree with the unsupervised labels
            idx_reduced_bool = preds[query_idx] != y_uns_label[query_idx]

            if sum(idx_reduced_bool) > 1:
                #If there is more than one prediction disagreement chose the first one
                idx_reduced = query_idx[idx_reduced_bool][0]
                idx_reduced = np.expand_dims(idx_reduced, axis=0)

            else:
                #If there is not disgreement just chose the first instance which is gotten by vote entropy
                print('No disagreement uns labels and pred labels')
                idx_reduced = query_idx[0]
                idx_reduced = np.expand_dims(idx_reduced, axis=0)
        else:
            #If no boostrapping use the most informative instance by vote entropy only
            idx_reduced = query_idx[0]
            idx_reduced = np.expand_dims(idx_reduced, axis=0)


        #For AMS data we need to query the user directly
        if baseline == False:
            #Check if the label has already been given and saved
            if np.isnan(y_pool_labels[idx_reduced]):

                print("Are this two listings the same one? 1-match , 0-nonmatch, 2-more=info")
                ids_int = [int(x) for x in ids[idx_reduced][0][1:-1].split(',')]
                print(compare(df_full,ids_int[0],ids_int[1]))
                label = int(input())
                #If label 2 we need more info
                if label == 2:
                    print(compare(df_full,ids_int[0],ids_int[1],full=True))
                    label = int(input())
                    y_new = np.array([label], dtype=float)
                else:
                    y_new = np.array([label], dtype=float)
                new_labels.append([(ids_int[0],ids_int[1]),label])
            #If label is already given just take it and dont ask for it again, saving time hopefully
            else:
                label = y_pool_labels[idx_reduced[0]]
                y_new = np.array([label], dtype=float)

        if baseline == True:
            #For baseline we can access the true labels no need to ask just retreive
            label = y_pool_labels[idx_reduced[0]]
            y_new = np.array([label], dtype=float)



        #This will add the labeled data into the same arrays
        X_train_lb = np.concatenate((X_train_lb,X_pool[idx_reduced]),axis=0)
        y_train_lb = np.concatenate((y_train_lb,y_new),axis =0)


        for model in committee:
            #Train the random forest in incremental way with the labeled instance
            if incremental_comitee == True:
                if type(model.get_params()['estimator'])== sklearn.ensemble._forest.RandomForestClassifier:

                    n_estimators = model.get_params()['estimator__n_estimators'] +2
                    params_rf = {'estimator__n_estimators':n_estimators} #,'estimator__max_depth': n_estimators
                    model.set_params(**params_rf)
                    #Teach the random forest , boostrap true is incremental
                    model.teach(X_train_lb, y_train_lb)

                    #Predict with the new instance
                    # we want to only use the prediction of the random forest which will be incrementally built
                else:

                    model.teach(X_train_lb, y_train_lb)
            else:
                model.teach(X_train_lb, y_train_lb)

        if warm == True and boostrap == True:
            #Teach the random forest Increase estimators for incremental learning
            model_main.n_estimators += 2
            #model_main.fit(X_pool_main,y_uns_label,sample_weight=y_uns_weight)
            model_main.fit(X_train_lb,y_train_lb,sample_weight =np.ones(y_train_lb.shape[0]))
        if warm == True and boostrap == False:
            #Teach the random forest Increase estimators for incremental learning
            model_main.n_estimators += 2
            #model_main.fit(X_pool_main,y_uns_label,sample_weight=y_uns_weight)
            model_main.fit(X_train_lb,y_train_lb)

        if warm ==False:
            model_main.fit(X_train_lb,y_train_lb)
        #Predict with the new instance
        # we want to only use the prediction of the random forest which will be incrementally built
        if committee_pred == False:
            y_pred = model_main.predict(X_test)
        else:
            y_pred = committee.predict(X_test)

        #Get evaluation scores
        precision_recall_fscore=precision_recall_fscore_support(y_test,y_pred,average='binary',zero_division=0)
        precision_scores.append(precision_recall_fscore[0])
        recall_scores.append(precision_recall_fscore[1])
        f_score.append(precision_recall_fscore[2])
        #Save results for further analysis
        dict_results = {'precision_scores':precision_scores,'recall_scores':recall_scores,
                        'f_score':f_score}

        #Delete the queried instance from the pool
        X_pool = np.delete(X_pool, idx_reduced, axis=0)
        ids = np.delete(ids, idx_reduced, axis=0)
        y_uns_label = np.delete(y_uns_label, idx_reduced, axis=0)
        y_pool_labels = np.delete(y_pool_labels, idx_reduced, axis=0)

        #Save labels to be added for further runs
        if baseline ==False:
            new_labels.append([(ids_int[0],ids_int[1]),label])


    #Graph to check the learning process
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plt.title('Precision of your model')
        plt.plot(range(n_queries+1), precision_scores)
        #plt.scatter(range(n_queries+1), precision_scores)
        plt.xlabel('number of queries')
        plt.ylabel('Precision')
        display.display(plt.gcf())
        plt.close('all')
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plt.title('recall_scores of your model')
        plt.plot(range(n_queries+1), recall_scores)
        #plt.scatter(range(n_queries+1), recall_scores)
        plt.xlabel('number of queries')
        plt.ylabel('recall_scores')
        display.display(plt.gcf())
        plt.close('all')
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1)
        plt.title('f_score of your model')
        plt.plot(range(n_queries+1), f_score)
        #plt.scatter(range(n_queries+1), f_score)
        plt.xlabel('number of queries')
        plt.ylabel('f_score')
        display.display(plt.gcf())
        plt.close('all')
    return new_labels, dict_results