import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from ntl_baseline import baseline
from ntl_baseline.config import BaselineConfig
from ntl_baseline.constants import LABELS, ContractError


def test_pipeline_matches_official_method():
    model = baseline.build_pipeline(BaselineConfig())
    params = model.get_params()
    assert params["features__word_tfidf__ngram_range"] == (1, 2)
    assert params["features__word_tfidf__max_features"] == 25_000
    assert params["features__char_tfidf__ngram_range"] == (3, 5)
    assert params["features__char_tfidf__max_features"] == 35_000
    assert params["features__char_tfidf__analyzer"] == "char"
    assert params["features__char_tfidf__sublinear_tf"] is True
    assert params["features__word_tfidf__sublinear_tf"] is True
    assert params["clf__C"] == 2.0
    assert params["clf__max_iter"] == 1000
    assert params["clf__solver"] == "lbfgs"
    assert params["clf__class_weight"] is None


def test_folds_match_official_groupkfold_without_prompt_overlap(train):
    config = BaselineConfig()
    folds = baseline.make_folds(train, config)
    np.testing.assert_array_equal(folds, baseline.make_folds(train, config))
    for fold, (fitting, heldout) in enumerate(
        GroupKFold(n_splits=5).split(train["text"], train["model"], train["prompt_id"])
    ):
        assert set(train.iloc[fitting].prompt_id).isdisjoint(train.iloc[heldout].prompt_id)
        np.testing.assert_array_equal(np.flatnonzero(folds == fold), heldout)


def test_not_enough_prompt_groups_is_an_error(train):
    with pytest.raises(ContractError, match="group"):
        baseline.make_folds(train[train.prompt_id != "prompt-00"], BaselineConfig())


def test_each_fold_must_cover_all_candidates(train):
    train = train[~((train.prompt_id == "prompt-00") & (train.model == LABELS[0]))]
    with pytest.raises(ContractError, match="18"):
        baseline.make_folds(train, BaselineConfig())


def test_oof_is_complete_and_metric_matches_official(train, monkeypatch):
    real_builder = baseline.build_pipeline
    seen_groups = []

    def checked_builder(config):
        model = real_builder(config)
        real_fit = model.fit

        def checked_fit(texts, labels):
            result = real_fit(texts, labels)
            groups_in_train = {g for g in range(5) if any(f"holdoutmarker{g}" in t for t in texts)}
            vocab = model.named_steps["features"].transformer_list[0][1].vocabulary_
            for group in set(range(5)) - groups_in_train:
                assert f"holdoutmarker{group}" not in vocab
            seen_groups.append(groups_in_train)
            return result

        model.fit = checked_fit
        return model

    monkeypatch.setattr(baseline, "build_pipeline", checked_builder)
    oof, report = baseline.cross_validate(train, BaselineConfig())
    assert oof.id.tolist() == train.id.tolist()
    assert oof.id.is_unique
    assert oof.groupby("prompt_id").fold.nunique().max() == 1
    assert len(seen_groups) == 5
    probs = oof[[f"prob__{label}" for label in LABELS]].to_numpy()
    assert probs.shape == (len(train), 18)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert report["oof_balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(train.model, oof.prediction)
    )
    assert len(report["folds"]) == 5
    assert set(report["per_class"]) == set(LABELS)
    assert np.asarray(report["confusion_matrix"]).sum() == len(train)
