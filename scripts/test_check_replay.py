"""Synthetic positive/negative tests; no bearing data files are read."""
from io import StringIO
import json
import unittest

import numpy as np
import pandas as pd

from check_replay import (check_weights, flags, metric_rows, pair_rules,
                          validate_events, validate_predictions)


def fixture(gap=False):
    """Hand-specified 214-observation streams with a known gap at index 207."""
    times = pd.Series(pd.date_range("2000-01-01", periods=214, freq="10min"))
    if gap:
        times.loc[207:] += pd.Timedelta(minutes=30)
    observations, events = [], []
    for bearing in range(1, 5):
        y = np.arange(214) / 1000 + bearing / 10
        observations.append(pd.DataFrame(dict(run=2, bearing=bearing,
                                             observation_index=np.arange(214), time=times,
                                             y=y, live=True)))
        inputs = list(range(200, 207)) + [212, 213] if gap else list(range(200, 214))
        for i in inputs:
            base = dict(run=2, bearing=bearing, input_index=i, time=times.iloc[i],
                        mlp=float(y[i]), target_index=np.nan, target_time=pd.NaT,
                        y=np.nan, reason="", eligible_for_scoring=False, eligible_for_training=False)
            events.append(dict(base, event="predict", event_time=times.iloc[i], observation_index=i))
            if gap and i in range(201, 207):
                events.append(dict(base, event="discard", event_time=times.iloc[207],
                                   observation_index=207, reason="gap"))
            elif i >= 208:
                events.append(dict(base, event="discard", event_time=times.iloc[213],
                                   observation_index=213, reason="end_of_stream"))
            else:
                j = i + 6
                events.append(dict(base, event="release", event_time=times.iloc[j],
                                   observation_index=j, target_index=j, target_time=times.iloc[j],
                                   y=float(y[j]), eligible_for_scoring=True, eligible_for_training=True))
    events = pd.DataFrame(events)
    events["phase"] = events.event.ne("predict").astype(int)
    events = events.sort_values(["event_time", "phase", "bearing", "input_index"]).drop(columns="phase").reset_index(drop=True)
    events["event_id"] = np.arange(len(events))
    predictions = events.loc[events.event == "release",
                             ["run", "bearing", "input_index", "target_index", "time", "target_time", "mlp", "y"]].copy()
    predictions["late"] = predictions.bearing == 1
    return pd.concat(observations, ignore_index=True), events, predictions


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        self.observations, self.events, self.predictions = fixture()
        self.rules = pair_rules(self.observations)

    def test_valid_log_and_reference(self):
        actual = validate_predictions(self.predictions, self.rules)
        self.assertEqual(len(actual), 32)
        self.assertEqual(validate_events(self.events, self.rules, actual)["predictions"], 56)

    def test_csv_round_trip(self):
        events = pd.read_csv(StringIO(self.events.to_csv(index=False)))
        predictions = pd.read_csv(StringIO(self.predictions.to_csv(index=False)))
        actual = validate_predictions(predictions, self.rules)
        self.assertEqual(validate_events(events, self.rules, actual)["status"], "passed")

    def test_actual_label_time_is_checked(self):
        changed = self.events.copy()
        i = changed.index[changed.event == "release"][0]
        # Falsify both the reported target time and event time: observations win.
        changed.loc[i, ["event_time", "target_time"]] -= pd.Timedelta(minutes=1)
        changed["phase"] = changed.event.ne("predict").astype(int)
        changed = changed.sort_values(["event_time", "phase", "bearing", "input_index"]).drop(columns="phase").reset_index(drop=True)
        changed["event_id"] = np.arange(len(changed))
        with self.assertRaisesRegex(ValueError, "actual observation time"):
            validate_events(changed, self.rules, self.predictions)

    def test_release_before_prediction_at_same_timestamp(self):
        changed = self.events.copy()
        release = changed.index[changed.event == "release"][0]
        prediction = changed.index[(changed.event == "predict") & (changed.bearing == 1)
                                   & (changed.event_time == changed.loc[release, "event_time"])][0]
        changed.iloc[[prediction, release]] = changed.iloc[[release, prediction]].to_numpy()
        changed["event_id"] = np.arange(len(changed))
        with self.assertRaisesRegex(ValueError, "chronological"):
            validate_events(changed, self.rules, self.predictions)

    def test_future_outcome_in_prediction_is_rejected(self):
        changed = self.events.copy()
        changed.loc[0, "y"] = .4
        with self.assertRaisesRegex(ValueError, "future outcome"):
            validate_events(changed, self.rules, self.predictions)

    def test_wrong_bearing_label_is_rejected(self):
        changed = self.events.copy()
        i = changed.index[changed.event == "release"][0]
        changed.loc[i, "y"] += .1
        with self.assertRaisesRegex(ValueError, "this bearing"):
            validate_events(changed, self.rules, self.predictions)

    def test_pending_identity_includes_bearing(self):
        changed = self.events.copy()
        releases = changed.index[changed.event == "release"][:2]
        changed.loc[releases[1], "bearing"] = changed.loc[releases[0], "bearing"]
        with self.assertRaisesRegex(ValueError, "no pending"):
            validate_events(changed, self.rules, self.predictions)

    def test_gap_discards_and_history_restart(self):
        observations, events, predictions = fixture(gap=True)
        rules = pair_rules(observations)
        self.assertEqual(set(rules.loc[rules.bearing == 1, "input_index"]),
                         set(range(200, 207)) | {212, 213})
        self.assertEqual(len(validate_predictions(predictions, rules)), 4)
        self.assertEqual(validate_events(events, rules, predictions)["predictions"], 36)
        changed = events.copy()
        i = changed.index[changed.reason == "gap"][0]
        changed.loc[i, "event"] = "release"
        changed.loc[i, "reason"] = ""
        with self.assertRaisesRegex(ValueError, "acquisition gap"):
            validate_events(changed, rules, predictions)

    def test_horizon_tolerance(self):
        changed = self.observations.copy()
        # Each interval is still regular (11 min), but six intervals are 66 min.
        changed["time"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(changed.observation_index * 11, unit="min")
        rules = pair_rules(changed)
        self.assertFalse(rules.eligible_for_training.any())
        self.assertTrue((rules.loc[rules.input_index < 208, "reason"] == "horizon").all())

    def test_missing_events_and_duplicates_fail(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_events(self.events.iloc[:-1], self.rules, self.predictions)
        duplicate = pd.concat([self.predictions, self.predictions.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_predictions(duplicate, self.rules)

    def test_eligibility_and_late_flags_are_recomputed(self):
        changed = self.events.copy()
        i = changed.index[changed.event == "release"][0]
        changed.loc[i, "eligible_for_training"] = False
        with self.assertRaisesRegex(ValueError, "eligible_for_training"):
            validate_events(changed, self.rules, self.predictions)
        changed = self.predictions.copy()
        changed.loc[changed.bearing == 2, "late"] = True
        with self.assertRaisesRegex(ValueError, "late"):
            validate_predictions(changed, self.rules)

    def test_changed_forecast_is_rejected(self):
        changed = self.events.copy()
        i = changed.index[changed.event == "release"][0]
        changed.loc[i, "mlp"] += .01
        with self.assertRaisesRegex(ValueError, "Forecast changed"):
            validate_events(changed, self.rules, self.predictions)

    def test_shutdown_is_scoring_filter_not_advance_knowledge(self):
        changed = self.observations.copy()
        changed.loc[changed.observation_index >= 212, "live"] = False
        rules = pair_rules(changed)
        last = rules.loc[rules.input_index == 207]
        self.assertTrue(last.eligible_for_training.all())
        self.assertFalse(last.eligible_for_scoring.any())
        self.assertEqual(len(rules), len(self.rules))

    def test_metrics_and_unavailable_late_subset(self):
        data = pd.DataFrame(dict(bearing=[1, 2, 3, 4], y=[0., 0., 0., 0.],
                                 mlp=[1., -2., 3., -4.], late=[True, False, False, False]))
        rows = metric_rows(data)
        self.assertEqual(rows[0]["mae"], 2.5)
        self.assertEqual(rows[0]["rmse"], np.sqrt(7.5))
        self.assertEqual(rows[0]["n_late"], 1)
        self.assertEqual(rows[0]["late_mae"], 1.)
        self.assertIsNone(rows[2]["late_mae"])
        json.dumps(rows, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "No scored"):
            metric_rows(data.iloc[:0])

    def test_changed_weights_are_rejected(self):
        original = {"W": np.zeros((2, 2))}
        changed = {"W": original["W"].copy()}
        check_weights(changed, original)
        changed["W"][0, 0] = 1.
        with self.assertRaisesRegex(ValueError, "Frozen weights changed"):
            check_weights(changed, original)

    def test_invalid_booleans_and_fractional_indices(self):
        with self.assertRaisesRegex(ValueError, "Invalid boolean"):
            flags(pd.Series(["False", "missing"]))
        changed = self.predictions.copy()
        changed["input_index"] = changed.input_index.astype(float) + .5
        with self.assertRaisesRegex(ValueError, "Invalid integer"):
            validate_predictions(changed, self.rules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
