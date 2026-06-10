# Data Transformations

The first training dataset should model the 90-minute match result. Extra time and penalties are useful metadata, but they should not overwrite the 90-minute score.

## Raw Inputs

- World Cup match files: `data/raw/openfootball-worldcup/worldcup-master/*/cup.txt`
- World Cup knockout files: `data/raw/openfootball-worldcup/worldcup-master/*/cup_finals.txt`
- FIFA ranking snapshots: `data/raw/fifa-rankings/by_schedule/*.json`

## Match Table

Create one row per match with:

- `tournament_year`
- `game_type`: initially always `championship`
- `stage_raw`
- `stage`: normalized value such as `group`, `round_of_16`, `quarter_final`, `semi_final`, `third_place`, `final`
- `is_knockout`
- `match_date`
- `country_a`
- `country_b`
- `country_a_code`
- `country_b_code`
- `goals_a_90`
- `goals_b_90`
- `goals_a_extra_time`
- `goals_b_extra_time`
- `penalties_a`
- `penalties_b`
- `venue`
- `host_country`
- `is_host_a`
- `is_host_b`

## Labels

Use these labels for the first models:

- `outcome_90`: `A_WIN`, `DRAW`, or `B_WIN`
- `goals_a_90`
- `goals_b_90`

Do not include an `outcome_final` label in the first pass.

## Ranking Features

Join each match to the ranking snapshot selected for its tournament year:

- `rank_a`
- `rank_b`
- `ranking_points_a`
- `ranking_points_b`
- `rank_diff`: `rank_b - rank_a`, positive when team A has the better rank
- `ranking_points_diff`: `ranking_points_a - ranking_points_b`
- `confederation_a`
- `confederation_b`
- `same_confederation`

Ranking snapshots are available for World Cups from 1994 onward. Earlier tournaments can either omit ranking features or use a missing-value strategy.

## Backtesting

Use rolling tournament windows. A first fair setup:

- Train: 1994, 1998, 2002, 2006. Test: 2010.
- Train: 1998, 2002, 2006, 2010. Test: 2014.
- Train: 2002, 2006, 2010, 2014. Test: 2018.
- Train: 2006, 2010, 2014, 2018. Test: 2022.

This keeps ranking features historically aligned and avoids using future tournament results for training.
