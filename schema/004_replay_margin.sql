-- How much cheaper than the current config a config had to measure to be
-- recommended, per replay run. Runs recorded before this column existed had
-- no margin: their cheapest config won by any amount.

ALTER TABLE replay_runs
    ADD COLUMN IF NOT EXISTS min_saving Float32 DEFAULT 0;
