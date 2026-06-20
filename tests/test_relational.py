"""Unit tests for Phase 1 relational feature engineering.

These use tiny hand-built frames with known answers, so the aggregation logic is
validated without the full ~690MB Home Credit dataset.
"""
import pandas as pd
import pytest

from credit_risk.features import relational as R


def _by_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("SK_ID_CURR")


def test_aggregate_bureau_basic_and_balance():
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
        "AMT_CREDIT_SUM_DEBT": [100.0, 0.0, 50.0],
        "AMT_CREDIT_SUM": [200.0, 200.0, 100.0],
        "CREDIT_DAY_OVERDUE": [5, 0, 0],
        "AMT_CREDIT_SUM_OVERDUE": [10.0, 0.0, 0.0],
        "DAYS_CREDIT": [-100, -200, -50],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [10, 10, 11],
        "MONTHS_BALANCE": [-1, -2, -1],
        "STATUS": ["1", "0", "C"],
    })
    out = _by_id(R.aggregate_bureau(bureau, bureau_balance))

    assert out.loc[1, "BUREAU_N_CREDITS"] == 2
    assert out.loc[1, "BUREAU_N_ACTIVE"] == 1
    assert out.loc[1, "BUREAU_ACTIVE_RATIO"] == pytest.approx(0.5)
    assert out.loc[1, "BUREAU_DEBT_SUM"] == pytest.approx(100.0)
    assert out.loc[1, "BUREAU_DEBT_MEAN"] == pytest.approx(50.0)
    assert out.loc[1, "BUREAU_OVERDUE_MAX"] == 5
    assert out.loc[1, "BUREAU_DAYS_CREDIT_MEAN"] == pytest.approx(-150.0)
    # bureau_balance: bureau 10 has one past-due month ("1"), bureau 11 none.
    assert out.loc[1, "BUREAU_BB_DPD_MONTHS_SUM"] == pytest.approx(1.0)
    assert out.loc[2, "BUREAU_ACTIVE_RATIO"] == pytest.approx(1.0)


def test_aggregate_bureau_without_balance():
    bureau = pd.DataFrame({"SK_ID_CURR": [1], "CREDIT_ACTIVE": ["Active"]})
    out = _by_id(R.aggregate_bureau(bureau, None))
    assert out.loc[1, "BUREAU_N_CREDITS"] == 1
    assert "BUREAU_BB_DPD_MONTHS_SUM" not in out.columns  # gracefully absent


def test_aggregate_previous_apps():
    prev = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "AMT_APPLICATION": [100.0, 100.0, 200.0],
        "AMT_CREDIT": [90.0, 0.0, 200.0],
        "DAYS_DECISION": [-10, -20, -5],
    })
    out = _by_id(R.aggregate_previous_apps(prev))
    assert out.loc[1, "PREV_N_APPS"] == 2
    assert out.loc[1, "PREV_APPROVED_RATIO"] == pytest.approx(0.5)
    assert out.loc[1, "PREV_REFUSED_RATIO"] == pytest.approx(0.5)
    assert out.loc[1, "PREV_AMT_CREDIT_MEAN"] == pytest.approx(45.0)
    assert out.loc[1, "PREV_CREDIT_TO_APP_RATIO"] == pytest.approx(0.45)
    assert out.loc[1, "PREV_DAYS_DECISION_MEAN"] == pytest.approx(-15.0)


def test_aggregate_installments_lateness_and_shortfall():
    inst = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "DAYS_INSTALMENT": [-30, -60],
        "DAYS_ENTRY_PAYMENT": [-25, -60],   # first paid 5 days late, second on time
        "AMT_INSTALMENT": [100.0, 100.0],
        "AMT_PAYMENT": [100.0, 80.0],        # second short by 20%
    })
    out = _by_id(R.aggregate_installments(inst))
    assert out.loc[1, "INST_N_INSTALMENTS"] == 2
    assert out.loc[1, "INST_LATE_RATE"] == pytest.approx(0.5)
    assert out.loc[1, "INST_DPD_MEAN"] == pytest.approx(2.5)
    assert out.loc[1, "INST_DPD_MAX"] == pytest.approx(5.0)
    assert out.loc[1, "INST_SHORTFALL_MEAN"] == pytest.approx(0.1)


def test_aggregate_pos():
    pos = pd.DataFrame({"SK_ID_CURR": [1, 1, 1], "SK_DPD": [0, 3, 0]})
    out = _by_id(R.aggregate_pos(pos))
    assert out.loc[1, "POS_N_MONTHS"] == 3
    assert out.loc[1, "POS_DPD_MEAN"] == pytest.approx(1.0)
    assert out.loc[1, "POS_DPD_MAX"] == 3
    assert out.loc[1, "POS_N_LATE_MONTHS"] == 1


def test_aggregate_credit_card_utilisation():
    cc = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "AMT_BALANCE": [50.0, 100.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 100.0],
        "SK_DPD": [0, 2],
    })
    out = _by_id(R.aggregate_credit_card(cc))
    assert out.loc[1, "CC_N_MONTHS"] == 2
    assert out.loc[1, "CC_BALANCE_MEAN"] == pytest.approx(75.0)
    assert out.loc[1, "CC_UTILIZATION_MEAN"] == pytest.approx(0.75)
    assert out.loc[1, "CC_DPD_MEAN"] == pytest.approx(1.0)


def test_build_relational_features_join_and_fill():
    app = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0]})
    bureau_agg = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "BUREAU_N_CREDITS": [2, 1],
        "BUREAU_ACTIVE_RATIO": [0.5, 1.0],
    })
    out = _by_id(R.build_relational_features(app, {"bureau": bureau_agg}))
    assert out.loc[3, "BUREAU_N_CREDITS"] == 0          # count miss -> 0
    assert pd.isna(out.loc[3, "BUREAU_ACTIVE_RATIO"])   # ratio miss -> NaN
    assert out.loc[1, "BUREAU_N_CREDITS"] == 2
    assert out.shape[0] == 3                             # left join keeps all apps


def test_missing_id_raises():
    with pytest.raises(ValueError):
        R.aggregate_bureau(pd.DataFrame({"X": [1]}))
