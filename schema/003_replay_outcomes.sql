-- What the replay lab records beyond the numbers of a tested config.

ALTER TABLE replay_runs
    ADD COLUMN IF NOT EXISTS note String;

-- outcome: tested, ruled_out (by the roofline check, before any restart),
-- failed_to_start, or skipped (the time budget ran out).
-- quality_unverified: the config changes precision, so it cannot win until
-- answer quality is checked.
ALTER TABLE replay_results
    ADD COLUMN IF NOT EXISTS outcome LowCardinality(String) DEFAULT 'tested',
    ADD COLUMN IF NOT EXISTS note String,
    ADD COLUMN IF NOT EXISTS quality_unverified Bool,
    ADD COLUMN IF NOT EXISTS baseline Bool,
    ADD COLUMN IF NOT EXISTS winner Bool;
